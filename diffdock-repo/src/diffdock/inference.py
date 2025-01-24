import fire
import functools
import logging
import pprint
import traceback
from argparse import ArgumentParser, Namespace, FileType
import copy
import os
import sys
import json
from pathlib import Path
from functools import partial
import warnings
from typing import Mapping, Optional, List
from types import SimpleNamespace

import yaml

# Ignore pandas deprecation warning around pyarrow
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="(?s).*Pyarrow will become a required dependency of pandas.*")

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs

# TODO imports are a little odd, utils seems to shadow things
from diffdock.utils.logging_utils import configure_logger, get_logger
import diffdock.utils.utils
from diffdock.datasets.process_mols import write_mol_with_coords
from diffdock.utils.download import download_and_extract
from diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from diffdock.utils.inference_utils import InferenceDataset, set_nones
from diffdock.utils.sampling import randomize_position, sampling
from diffdock.utils.utils import get_model
from diffdock.utils.visualise import PDBFile
from tqdm import tqdm

if os.name != 'nt':  # The line does not work on Windows
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`")

# Prody logging is very verbose by default
prody_logger = logging.getLogger(".prody")
prody_logger.setLevel(logging.ERROR)

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]

class Inference:
    def __init__(
            self,
            confidence_model_dir: str=None,
            config: str="default_inference_args.yaml",
            loglevel: str="WARNING",
            out_dir: str="results/user_inference",
            save_visualisation: bool=False,
            samples_per_complex: int=10,
            model_dir: str=None,
            ckpt: str="best_ema_inference_epochS_model.pt",
            confidence_ckpt: str="best_model.py",
            batch_size: int=10,
            no_final_step_noise: bool=True,
            inference_steps: int=20,
            actual_steps: int=None,
            old_score_model: bool=False,
            old_confidence_model: bool=True,
            initial_noise_std_proportion: float=-1.0,
            choose_residue: bool=False,
            temp_sampling_tr: float=1.0,
            temp_psi_tr: float=0.0,
            temp_sigma_data_tr: float=0.5,
            temp_sampling_rot: float=1.0,
            temp_psi_rot: float=0.0,
            temp_sigma_data_rot: float=0.5,
            temp_sampling_tor: float=1.0,
            temp_psi_tor: float=0.0,
            temp_sigma_data_tor: float=0.5,
            gnina_minimize: bool=False,
            gnina_path: str="gnina",
            gnina_log_file: str="gnina_log.txt",
            gnina_full_dock: bool=False,
            gnina_autobox_add:float=4.0,
            gnina_poses_to_optimize: int=1):
        """
        Partial list of input argument annotations:
        out_dir:
          Directory where the outputs will be written to
        save_visualization:
          Save a pdb file with all of the steps of the reverse diffusion
        samples_per_complex:
          Number of samples to generate
        model_dir:
          Path to folder with trained score model and hyperparameters
        ckpt:
          Checkpoint to use for the score model
        confidence_model_dir:
          Path to folder with trained confidence model and hyperparameters
        confidence_ckpt:
          Checkpoint to use for the confidence model
        no_final_step_noise:
          Use no noise in the final step of the reverse diffusion
        inference_steps:
          Number of denoising steps
        actual_steps:
          Number of denoising steps that are actually performed
        initial_noise_std_proportion:
          Initial noise std proportion
        """
        args = SimpleNamespace(**locals())
        if config:
            with open(config, 'r') as fh:
                config_dict = yaml.load(fh, Loader=yaml.FullLoader)
                arg_dict = args.__dict__
                for key, value in config_dict.items():
                    if isinstance(value, list):
                        for v in value:
                            arg_dict[key].append(v)
                    else:
                        arg_dict[key] = value

        self.__dict__.update(args.__dict__)
        configure_logger(loglevel)
        self.logger = get_logger()

        os.makedirs(args.out_dir, exist_ok=True)
        with open(f'{args.model_dir}/model_parameters.yml') as f:
            score_model_args = Namespace(**yaml.full_load(f))
        if args.confidence_model_dir is not None:
            with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
                confidence_args = Namespace(**yaml.full_load(f))

        self.score_model_args = score_model_args
        self.confidence_args = confidence_args
        self.t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

        # preprocessing of complexes into geometric graphs
        self.test_dataset = InferenceDataset(
                out_dir=args.out_dir,
                receptor_radius=score_model_args.receptor_radius,
                remove_hs=score_model_args.remove_hs,
                c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                all_atoms=score_model_args.all_atoms,
                atom_radius=score_model_args.atom_radius,
                atom_max_neighbors=score_model_args.atom_max_neighbors,
                knn_only_graph=(
                    False if not hasattr(score_model_args, 'not_knn_only_graph') 
                    else not score_model_args.not_knn_only_graph))

        self.confidence_test_dataset = None
        if (args.confidence_model_dir is not None and not
            confidence_args.use_original_model_cache):
            self.logger.info(
                    'Confidence model uses different type of graphs than the score model. '
                    'Loading (or creating if not existing) the data for the confidence model now.')
            self.confidence_test_dataset = \
                    InferenceDataset(
                            out_dir=args.out_dir, 
                            receptor_radius=confidence_args.receptor_radius,
                            remove_hs=confidence_args.remove_hs,
                            c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                            all_atoms=confidence_args.all_atoms,
                            atom_radius=confidence_args.atom_radius,
                            atom_max_neighbors=confidence_args.atom_max_neighbors,
                            knn_only_graph=(
                                False if not hasattr(score_model_args, 'not_knn_only_graph')
                                else not score_model_args.not_knn_only_graph
                                ))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(score_model_args, self.device, t_to_sigma=self.t_to_sigma,
                          no_parallel=True, old=args.old_score_model)
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', 
                                map_location=torch.device('cpu'),
                                weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        self.model = model

        if args.confidence_model_dir is not None:
            confidence_model = get_model(confidence_args, self.device,
                                         t_to_sigma=self.t_to_sigma, no_parallel=True,
                                         confidence_mode=True,
                                         old=args.old_confidence_model)
            state_dict = torch.load(
                    f'{args.confidence_model_dir}/{args.confidence_ckpt}', 
                    map_location=torch.device('cpu'),
                    weights_only=True)
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(self.device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None

        self.confidence_model = confidence_model


    def main(self,
             complex_names: List[str],
             protein_paths: List[str],
             ligand_descriptions: List[str]):

        # print(json.dumps(args.__dict__, indent=2))

        # Download models if they don't exist locally
        """
        if not os.path.exists(args.model_dir):
            logger.info(f"Models not found. Downloading")
            remote_urls = REMOTE_URLS
            downloaded_successfully = False
            for remote_url in remote_urls:
                try:
                    logger.info(f"Attempting download from {remote_url}")
                    files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
                    if not files_downloaded:
                        logger.info(f"Download from {remote_url} failed.")
                        continue
                    logger.info(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
                    downloaded_successfully = True
                    # Once we have downloaded the models, we can break the loop
                    break
                except Exception as e:
                    pass

            if not downloaded_successfully:
                raise Exception(f"Models not found locally and failed to download them from {remote_urls}")
        """

        self.logger.info(f"DiffDock will run on {self.device}")

        for name in complex_names:
            write_dir = f'{self.out_dir}/{name}'
            os.makedirs(write_dir, exist_ok=True)

        self.test_dataset.initialize(complex_names, protein_paths, ligand_descriptions)
        self.test_dataset.compute_lm_embeddings()

        test_loader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        # t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
        tr_schedule = get_t_schedule(inference_steps=self.inference_steps,
                                     sigma_schedule='expbeta')

        failures, skipped = 0, 0
        N = self.samples_per_complex
        test_ds_size = len(self.test_dataset)
        self.logger.info(f'Size of test dataset: {test_ds_size}')
        for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
            if not orig_complex_graph.success[0]:
                skipped += 1
                self.logger.warning(
                        f"The test dataset did not contain "
                        f"{self.test_dataset.complex_names[idx]} for "
                        f"{self.test_dataset.ligand_descriptions[idx]} and "
                        f"{self.test_dataset.protein_paths[idx]}. "
                        f"We are skipping this complex.")
                continue
            try:
                if self.confidence_test_dataset is not None:
                    self.confidence_test_dataset.initialize(complex_names,
                                                            protein_paths,
                                                            ligand_descriptions)
                    self.confidence_test_dataset.set_lm_embeddings(self.test_dataset.lm_embeddings)
                    confidence_complex_graph = self.confidence_test_dataset[idx]
                    if not confidence_complex_graph.success:
                        skipped += 1
                        self.logger.warning(f"The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                        continue
                    confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
                else:
                    confidence_data_list = None
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                randomize_position(data_list, self.score_model_args.no_torsion, False,
                                   self.score_model_args.tr_sigma_max,
                                   initial_noise_std_proportion=self.initial_noise_std_proportion,
                                   choose_residue=self.choose_residue)

                lig = orig_complex_graph.mol[0]

                # initialize visualisation
                pdb = None
                if self.save_visualisation:
                    visualization_list = []
                    for graph in data_list:
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                # run reverse diffusion
                data_list, confidence = sampling(
                        data_list=data_list, model=self.model,
                        inference_steps=self.actual_steps if self.actual_steps is not
                        None else self.inference_steps, tr_schedule=tr_schedule,
                        rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                        device=self.device, t_to_sigma=self.t_to_sigma,
                        model_args=self.score_model_args,
                        visualization_list=visualization_list,
                        confidence_model=self.confidence_model,
                        confidence_data_list=confidence_data_list,
                        confidence_model_args=self.confidence_args,
                        batch_size=self.batch_size,
                        no_final_step_noise=self.no_final_step_noise,
                        temp_sampling=[self.temp_sampling_tr, self.temp_sampling_rot,
                                       self.temp_sampling_tor],
                        temp_psi=[self.temp_psi_tr, self.temp_psi_rot, self.temp_psi_tor],
                        temp_sigma_data=[self.temp_sigma_data_tr,
                                         self.temp_sigma_data_rot,
                                         self.temp_sigma_data_tor])

                ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() +
                                         orig_complex_graph.original_center.cpu().numpy()
                                         for complex_graph in data_list])

                # reorder predictions based on confidence output
                if confidence is not None and isinstance(
                        self.confidence_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    confidence = confidence[re_order]
                    ligand_pos = ligand_pos[re_order]

                # save predictions
                write_dir = f'{self.out_dir}/{complex_names[idx]}'
                for rank, pos in enumerate(ligand_pos):
                    mol_pred = copy.deepcopy(lig)
                    if self.score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)
                    if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                    write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

                # save visualisation frames
                if self.save_visualisation:
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
                    else:
                        for rank, batch_idx in enumerate(ligand_pos):
                            visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

            except Exception as e:
                self.logger.warning("Failed on", orig_complex_graph["name"], e)
                failures += 1

        result_msg = f"""
        Failed for {failures} / {test_ds_size} complexes.
        Skipped {skipped} / {test_ds_size} complexes.
        """
        if failures or skipped:
            self.logger.warning(result_msg)
        else:
            self.logger.info(result_msg)
        self.logger.info(f"Results saved in {self.out_dir}")

def batch_inputs(inputs: List, batch_size: int):
    """
    inputs: List of dicts of  
    """
    inputs = [tuple(inputs[i:i+batch_size]) for i in range(0, len(inputs), batch_size)]
    batched = []
    for group in inputs:
        entry = { k + 's': [] for k in group[0].keys() }
        for ent in group:
            for k, v in ent.items():
                entry[k+'s'].append(v)
        batched.append(entry)
    return batched

def run_batch(hps: str, inputs: str, batch_size: int=10):
    """
    hps: File with JSON-encoded hyperparams, see data/hps.json
    inputs: File with JSON-encoded inputs, see data/dockgen.json
    """
    hps = json.loads(Path(hps).read_text())
    inputs = json.loads(Path(inputs).read_text())
    batches = batch_inputs(inputs, 10)
    inf = Inference(**hps)
    for batch in batches:
        inf.main(**batch)

if __name__ == "__main__":
    fire.Fire(run_batch)


import modal
from pathlib import Path
import os
import json

repo = Path(__file__).parent / "diffdock-repo"

VOLUME_DIR = Path("/app")
CACHE_DIR = VOLUME_DIR / "cache"
VOLUME_NAME = os.environ.get("VOLUME_NAME")
if VOLUME_NAME is None:
    raise RuntimeError(f"Please set environment variable 'VOLUME_NAME'")

CONCURRENCY_LIMIT=2

volume = modal.Volume.from_name("diffdock-vol", create_if_missing=True)
app = modal.App()
image = (
        modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel", 
                                  add_python="3.11")
        .apt_install("git")
        .run_commands("pip install fair-esm[esmfold]==2.0.0 --no-deps")
        .pip_install_from_pyproject(
            repo / "pyproject.toml", 
            find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html")
        .env({'TORCH_HOME': str(CACHE_DIR)})
        .add_local_dir(repo / "src/diffdock", "/root/diffdock")
        .add_local_file(repo / "data/hps.json", "/root/hps.json")
        )


@app.cls(gpu="H100", image=image, volumes={VOLUME_DIR: volume}, 
         concurrency_limit=CONCURRENCY_LIMIT)
class Model:
    @modal.enter()
    def on_startup(self):
        hps = json.loads(Path("/root/hps.json").read_text())
        os.chdir(VOLUME_DIR)
        from diffdock import inference, prepare
        prepare.load_caches(CACHE_DIR)
        self.ddif = inference.Inference(**hps)

    @modal.method()
    def dock(self, input: dict):
        """
        input: dict with keys pointing to parallel lists:
          - complex_names: List of string, arbitrary names
          - protein_paths: List of string, local paths to pdb file
          - ligand_descriptions: List of string, local paths to ligand .sdf file 
        """
        print(f"Processing {input['complex_names']}...")
        self.ddif.main(**input)
        return input["complex_names"]

@app.local_entrypoint()
def main(inputs_json: str, batch_size: int):
    """
    inputs: JSON file describing input protein-ligand pairs (see diffdock.prepare)
    batch_size: number of protein-ligand pairs to submit for each job
    """
    inputs = json.loads(Path(inputs_json).read_text())
    batched = batch_inputs(inputs, batch_size)
    model = Model()
    print("finished instantiating Model class")
    # outputs only written to volume (for now)
    for result in model.dock.map(batched, order_outputs=False):
        print(result)

@app.function(image=image, volumes={VOLUME_DIR: volume}, concurrency_limit=1)
def download_models():
    from diffdock import prepare
    prepare.download_models(VOLUME_DIR)

@app.function(image=image, volumes={VOLUME_DIR: volume}, concurrency_limit=1)
def build_caches():
    from diffdock import prepare
    prepare.build_caches(str(CACHE_DIR))


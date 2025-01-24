import modal
from pathlib import Path
from typing import List, Dict
import os
import sys
import signal
import json

def handle_sigquit(signum, frame):
    print("Received SIGQUIT. Printing stack trace:")
    import traceback
    traceback.print_stack(frame)
    sys.exit(1)

# Register the handler
signal.signal(signal.SIGQUIT, handle_sigquit)
        
def batch_inputs(inputs: List, batch_size: int):
    inputs = [tuple(inputs[i:i+batch_size]) for i in range(0, len(inputs), batch_size)]
    batched = []
    for group in inputs:
        entry = { k + 's': [] for k in group[0].keys() }
        for ent in group:
            for k, v in ent.items():
                entry[k+'s'].append(v)
        batched.append(entry)
    return batched

here = Path(__file__).parent


app = modal.App()
image = (
        modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel", 
                                  add_python="3.11")
        .apt_install("git")
        .run_commands(
            "pip install fair-esm[esmfold]==2.0.0 --no-deps"
            )
        .pip_install_from_pyproject(
            here / "pyproject.toml", 
            find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html")
        .env({'TORCH_HOME': "/app/cache"})
        # must be last (to get nice mounting behavior)
        # this allows `diffdock` package to be imported without pip install, 
        # since Modal adds /root to sys.path
        .add_local_dir(here / "src/diffdock", "/root/diffdock")
        # .add_local_file(here / "data/dockgen.json", "/root/")
        # .add_local_file(here / "data/hps.json", "/root/")
        )

# global file paths
# HPS_PATH = Path("/root/hps.json")
# INPUTS_PATH = Path("/root/dockgen.json")

volume = modal.Volume.from_name("diffdock-vol", create_if_missing=True)
MODEL_DIR = Path("/app")

@app.cls(gpu="H100", image=image, volumes={MODEL_DIR: volume}, concurrency_limit=2)
class Model:
    @modal.enter()
    def on_startup(self):
        self.hps = json.loads(HPS_PATH.read_text())
        os.chdir(MODEL_DIR)
        from diffdock import inference
        self.ddif = inference.Inference(**self.hps)

    @modal.method()
    def dock(self, input: dict):
        """
        input: dict with keys pointing to parallel lists:
          - complex_names: List of string, arbitrary names
          - protein_paths: List of string, local paths to pdb file
          - ligand_descriptions: List of string, local paths to ligand .sdf file 
        """
        # help(inference)
        print(f"Processing {input['complex_names']}...")
        self.ddif.main(**input)
        return input["complex_names"]

@app.local_entrypoint()
def main(hps: str, inputs: str, batch_size: int):
    """
    hps: JSON file with kwargs to inference.py::main function
    inputs: JSON file describing input protein-ligand pairs (see diffdock.prepare)
    batch_size: number of protein-ligand pairs to submit for each job
    """
    inputs = json.loads(Path(inputs).read_text())
    hps = json.loads(Path(hps).read_text())
    batched = batch_inputs(inputs, batch_size)

    # model = Model()
    # outputs only written to volume (for now)
    # for result in model.dock.map(batched, order_outputs=False):
     #    print(result)



@app.function(gpu="H100", image=image, volumes={MODEL_DIR: volume}, concurrency_limit=2)
# @app.function(image=image, volumes={MODEL_DIR: volume}, concurrency_limit=2)
def dock(input: dict, *, hps: dict):
    """
    hps: dict of hyperparameters 
    input: dict with keys pointing to parallel lists:
      - complex_names: List of string, arbitrary names
      - protein_paths: List of string, local paths to pdb file
      - ligand_descriptions: List of string, local paths to ligand .sdf file 
    """
    os.chdir(MODEL_DIR)
    from diffdock import inference
    ddif = inference.Inference(**hps)
    # help(inference)
    print(f"Processing {input['complex_names']}...")
    ddif.main(**input)
    return input["complex_names"]

def _leaktest():
    # run some ablation of dock function (ablating code in inference.py) to see
    # what is responsible for the CPU memory leak
    """
    When I run this using:
    laptop $ modal shell app.py
    container $ python
    container $ >>> import app
    container $ >>> app._leaktest() 

    I get this warning, but I do have access to the mounted `diffdock-vol`:
    /pkg/modal/functions.py:1382: 
      UserWarning: The dock function is executing locally and will not have access to
      the mounted Volume or NetworkFileSystem data
    """
    import resource
    hps = json.loads(Path("/root/hps.json").read_text())
    inputs = json.loads(Path("/root/dockgen.json").read_text())
    batched = batch_inputs(inputs, 10)
    for batch in batched[:10]:
        dock.local(batch, hps=hps)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(usage.ru_maxrss)

import modal
from pathlib import Path
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
        .add_local_dir(here / "src/diffdock", "/root/diffdock")
        )
"""
.run_commands(
    "pip install "
    '"DiffDock @ git+https://github.com/hrbigelow/DiffDock.git@modal-port" '
    "--find-links 
    # force_build = True
    )
"""
        # )

volume = modal.Volume.from_name("diffdock-vol", create_if_missing=True)
MODEL_DIR = Path("/app")

@app.function(gpu="H100", image=image, volumes={MODEL_DIR: volume},
              concurrency_limit=2)
def dock(input: dict, *, hps: dict):
    """
    hps: serialized JSON object with kwargs to inference.py::main function
    input: serialized JSON object with keys:
      - complex_name: string, arbitrary name
      - protein_path: string, local path to pdb file
      - ligand_description: string, local path to ligand .sdf file 
    """
    os.chdir(MODEL_DIR)
    from diffdock import inference
    inference.main(**hps, **input)
    return input["complex_name"]

@app.local_entrypoint()
def main(hps: str, inputs: str):
    """
    hps: serialized JSON object with kwargs to inference.py::main function
    inputs: path to ...
    """
    hps = json.loads(Path(hps).read_text())
    inputs = json.loads(Path(inputs).read_text())

    # outputs only written to volume (for now)
    for result in dock.map(inputs, kwargs=dict(hps=hps), order_outputs=False):
        print(result)


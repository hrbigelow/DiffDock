import modal
from pathlib import Path
import os
import sys
import signal

def handle_sigquit(signum, frame):
    print("Received SIGQUIT. Printing stack trace:")
    import traceback
    traceback.print_stack(frame)
    sys.exit(1)

# Register the handler
signal.signal(signal.SIGQUIT, handle_sigquit)

app = modal.App()
image = (
        modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel", 
                                  add_python="3.11")
        .apt_install("git")
        .run_commands(
            "pip install fair-esm[esmfold]==2.0.0 --no-deps"
            )
        .run_commands(
            "pip install "
            '"DiffDock @ git+https://github.com/hrbigelow/DiffDock.git@modal-port" '
            "--find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html",
            # force_build = True
            )
        .env({'TORCH_HOME': "/app/cache"})
        )

volume = modal.Volume.from_name("diffdock-vol", create_if_missing=True)
MODEL_DIR = Path("/app")

@app.function(gpu="A100", image=image, volumes={MODEL_DIR: volume})
def run():
    os.chdir(MODEL_DIR)
    from diffdock import inference
    kwargs = {
            'config': 'default_inference_args.yaml',
            'protein_ligand_csv': 'data/protein_ligand_example.csv',
            }
    inference.main(**kwargs)


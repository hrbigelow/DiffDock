# Porting DiffDock to Modal

## Initial preparation of DiffDock repo

### Convert DiffDock into a library package
  - change all absolute imports to relative
  - move code into `src/diffdock`
  - add pyproject.toml
  - move from argparse -> fire + plain kwargs main function

### Upgrade to CUDA 12.1 + torch 2.4.0
  - fix broken import fair-esm[esmfold] (use --no-deps)
  - use `-f https://data.pyg.org/whl/torch-2.4.0+cu121.html` for torch-scatter etc.


## modal-app.py
  - uses `pip_install_from_pyproject` with DiffDock `pyproject.toml`
  - uses `add_local_dir` to add the DiffDock source directory



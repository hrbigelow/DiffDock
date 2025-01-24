# Installation

    modal


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


## Runs 

At the first large-scale run (`--inputs data/dockgen.json`, `concurrency_limit=10`) 
there is apparently a gradual CPU memory leak (and then something more sudden):

<img src="./img/first_large_run.png"></img>

Next was to add batched dispatch.  Each invocation of 

<img src="./img/batch5_run1.png"></img>

Memory leak

```
>>> for batch in batched[:10]:
...     inference.main(**hps, **batch)
...     usage = resource.getrusage(resource.RUSAGE_SELF)
...     print(usage.ru_maxrss)
...

# shows:
11521272
11612920
11736568
11821816
11914492
11987964
12096764

sizes = [11521272, 11612920, 11736568, 11821816, 11914492, 11987964, 12096764]
>>> [s2 - s1 for s1, s2 in zip(sizes[:-1], sizes[1:])]
[91648, 123648, 85248, 92676, 73472, 108800]
```

### Changes to DiffDock code

I made two major changes to DiffDock code `inference.py` and
`utils/inference_utils.py`.  Both involved separating model instantiation +
checkpoint loading from model forward call code.  

instantiated models are now held in
`inference.py::Inference` and
`inference_utils.py::InferenceDataset`.  And, the code which prepares the models for
specific protein-ligand inputs is in `inference.py::Inference.main` and
`inference_utils::InferenceDataset.initialize`.

Secondly, I ported `app.py` to now use a `app.cls()`.  Here, instantiation of
`Inference()` class is done in `@modal.enter` function, and the `main` call in the
`@modal.method` function.  The leak is now solved:

<img src="./img/after_cls_refactor.png"></img>


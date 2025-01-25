from pathlib import Path
import itertools
import fire
import os
import json
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def input_json(source_root_path: str, 
               dest_rel_path: str, 
               out_file: str, 
               protein_suffix: str="_protein_processed.pdb", 
               ligand_suffix: str="_ligand.pdb"):
    """
    source_root_path: a path to a directory containing subdirectories, each containing
    exactly one file ending in `protein_suffix` and one or more files ending in
    `ligand_suffix`.  For such a subdirectory, one JSON array entry is created for
    each protein-ligand pair.

    dest_rel_path: relative path 
    """
    output = []
    for dirpath, dirnames, filenames in os.walk(source_root_path):
        protein_files = [f for f in filenames if f.endswith(protein_suffix)]
        ligand_files = [f for f in filenames if f.endswith(ligand_suffix)]
        tag = Path(dirpath).name
        for i, (p, l) in enumerate(itertools.product(protein_files, ligand_files)):
            entry = { 
             "complex_name": f"{tag}_{i}",
             "protein_path": str(Path(dest_rel_path) / tag / p),
             "ligand_description": str(Path(dest_rel_path) / tag / l)
             }
            output.append(entry)
    out_str = json.dumps(output, indent=2)
    Path(out_file).write_text(out_str)

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]

def download_models(model_dir: str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    remote_urls = REMOTE_URLS
    success = False
    for remote_url in remote_urls:
        try:
            print(f"Attempting download from {remote_url}")
            resp = urlopen(remote_url)
            with ZipFile(BytesIO(resp.read())) as zip_file:
                files_downloaded = zip_file.namelist()
                zip_file.extractall(model_dir)
            print(f"Extracted {len(files_downloaded)} files to {model_dir}")
            success = True
            break
        except Exception as e:
            pass

    if not success:
        raise RuntimeError(
                f"Models not found locally and failed to download them from {remote_urls}")

def build_caches(cache_dir: str):
    from diffdock.utils import so3, torus
    print(f"Building SO3 cache (this may take ~10 minutes) ...")
    so3.build_cache(cache_dir)
    print(f"Building SO(2)/torus cache (this may take awhile) ...")
    torus.build_cache(cache_dir)
    print(f"Wrote caches to {cache_dir}") 

def load_caches(cache_dir: str):
    from diffdock.utils import so3, torus
    so3.load_cache(cache_dir)
    torus.load_cache(cache_dir)

if __name__ == '__main__':
    cmds = dict(make_input=input_json, download=download_models)
    fire.Fire(cmds)







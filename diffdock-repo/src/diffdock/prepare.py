from pathlib import Path
import itertools
import fire
import os
import json

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

if __name__ == '__main__':
    fire.Fire(input_json)







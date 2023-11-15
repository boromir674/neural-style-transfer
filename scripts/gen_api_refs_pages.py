"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files


src = Path(__file__).parent.parent / "src"  

for path in sorted(src.rglob("*.py")):  
    module_path = path.relative_to(src).with_suffix("")  
    doc_path = path.relative_to(src).with_suffix(".md")  
    full_doc_path = Path("reference", doc_path)  

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  
        identifier = ".".join(parts)  
        print("::: " + identifier, file=fd)  


    # ROOT
    #  -> docs
    #     -> scripts
    # -> src/python_package
    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)
    # so that it correctly sets the edit path of (for example) nst_math.py to
    # <repo_url>/blob/master/src/artificial_artwork/nst_math.py instead of
    # <repo_url>/blob/master/docs/src/artificial_artwork/nst_math.py


    # mkdocs_gen_files.set_edit_path(full_doc_path, path)  
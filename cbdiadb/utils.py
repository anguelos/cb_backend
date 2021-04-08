from typing import List
import pathlib

def imagepath_to_pagenum(page_path:str):
    page_num = int(pathlib.Path(page_path).name.split(".")[0].split("!")[0].split("_")[-1])
    return page_num


def filepath_to_pageid(filepath: str, db_root: str):
    return documentpath_to_documentid(filepath, db_root), imagepath_to_pagenum(filepath)


def pageids_to_filepaths(filepaths:List[str], db_root:str):
    return {filepath_to_pageid(f, db_root): f for f in filepaths}


def documentpath_to_documentid(documentpath, document_root):
    path = pathlib.Path(documentpath)
    root_parts = pathlib.Path(document_root).parts
    assert path.exists()
    assert path.parts[:len(root_parts)] == root_parts
    if not path.is_dir():
        path = path.parent
    assert path.is_dir()
    return "/".join(path.parts[len(root_parts):])

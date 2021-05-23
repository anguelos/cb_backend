from typing import List, Tuple
import pathlib
import os


def imagepath_to_pagenum(page_path: str) -> int:
    page_num = int(pathlib.Path(page_path).name.split(".")[0].split("!")[0].split("_")[-1])
    return page_num


def filepath_to_pageid(filepath, db_root: str = "", name_depth=-1) -> Tuple[str, int]:
    if db_root == 0 and name_depth == -1:
        raise ValueError
    elif db_root != "" and name_depth == -1:
        return _filepath_to_pageid_root(filepath, db_root=db_root)
    elif db_root == "" and name_depth > -1:
        return _filepath_to_pageid_no_root(filepath, name_depth=name_depth)
    else:
        raise ValueError


def pageids_to_filepaths(filepaths:List[str], db_root: str = "", name_depth: int = -1) -> List[Tuple[str, int]]:
    return {filepath_to_pageid(f, db_root=db_root, name_depth=name_depth): f for f in filepaths}


def documentpath_to_documentid(documentpath: str, document_root: str) -> str:
    path = pathlib.Path(documentpath)
    root_parts = pathlib.Path(document_root).parts
    assert path.exists()
    assert path.parts[:len(root_parts)] == root_parts
    if not path.is_dir():
        path = path.parent
    assert path.is_dir()
    return "/".join(path.parts[len(root_parts):])


def _filepath_to_pageid_no_root(filepath: str, name_depth: int = 3) -> Tuple[str, int]:
    doc_ids = []
    path, name = os.path.split(filepath)
    doc_ids = []
    for n in range(name_depth):
        path, folder = os.path.split(path)
        doc_ids.insert(0, folder)
    return "/".join(doc_ids), imagepath_to_pagenum(name)


def _filepath_to_pageid_root(filepath: str, db_root: str) -> Tuple[str, int]:
    return documentpath_to_documentid(filepath, db_root), imagepath_to_pagenum(filepath)


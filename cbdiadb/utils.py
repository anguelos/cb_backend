from typing import List


def imagepath_to_pagenum(page_path:str):
    page_num = int(page_path.split("/")[-1].split(".")[0].split(".")[0].split("!")[0].split("_")[-1])
    return page_num


def filepath_to_pageid(filepath:str):
    return filepath.split("/")[-1].split(".")[0]


def pageids_to_filepaths(filepaths:List[str]):
    return {filepath_to_pageid(f): f for f in filepaths}
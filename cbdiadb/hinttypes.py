import torch
from io import BytesIO
from typing import Union, Tuple, List
from pathlib import Path
from PIL import Image
import numpy as np

t_ctx = List[str]

t_size = Tuple[int, int] # width height

t_size_or_none = Union[t_size, None]

t_rect = Tuple[int, int, int, int]

t_filename = Union[Path, str]
t_image = Union[Path, str, Image.Image, np.ndarray, BytesIO]
t_optional_tensor = Union[torch.Tensor, None]
t_optional_array = Union[np.ndarray, None]

t_embed_subject: Union[np.ndarray, str]

t_indexreply_ind = Tuple[np.ndarray, np.ndarray] # rect index, confidence
t_indexreply_rect = Tuple[List[str], np.ndarray] # rect_str, confidence

t_dbreply_rect = t_rect
t_dbreply_row = Tuple[float, str, int, t_dbreply_rect]

t_srvreply_row = Tuple[str, int, int, int, int, int]
t_srvquery_ctx = List[str]

t_srvquery_imgref = Tuple[str, int , t_rect]


import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from imageio import imread


def fix_seed(seed: int = 56) -> None:
    """
    Fix all random seeds for reproducibility
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_benchmark_image(
    type: str,
    name: str,
    generic_2d_mono_raw_folder: str = "ssi/benchmark/images/generic_2d_all",
):
    generic_2d_mono_raw_folder = Path(generic_2d_mono_raw_folder)
    folder = generic_2d_mono_raw_folder / type
    if not folder.exists():
        folder = Path("../..") / folder
    try:
        files = [f for f in folder.iterdir() if f.is_file()]
    except FileNotFoundError as e:
        print("File not found, cwd:", os.getcwd())
        raise e
    filename = [f.name for f in files if name in f.name][0]
    filepath = folder / filename
    array = imread(filepath)
    return array, filename


def print_score(header: str, val1: float, val2: float, val3: float, val4: float):
    print(f"| {header:30s} | {val1:.4f} | {val2:.4f} | {val3:.4f} | {val4:.4f} |")


def print_header(columns: List[str]):
    header = f"| {' ' * 30} | {' | '.join(columns)} |"
    separator = f"| {'-' * 30} | {' | '.join(['-' * len(c) for c in columns])} |"
    print(header)
    print(separator)

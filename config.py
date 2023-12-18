from dataclasses import dataclass

import torch


@dataclass
class Config:
    save_step: int = 5
    dataroot: str = "celeba/"
    batch_size: int = 128
    image_size: int = 64
    num_epochs: int = 5
    ngpu: int = 1
    nc: int = 3
    nz: int = 100
    ngf: int = 64
    ndf: int = 64
    lr: float = 0.0002
    beta1: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "celeba"
    workers: int = 2

from dataclasses import dataclass

import torch


@dataclass
class Config:
    # Model
    nc: int = 3
    nz: int = 100
    ngf: int = 64
    ndf: int = 64

    # Optimizer
    lr: float = 0.0002
    beta1: float = 0.5

    # Dataloader
    dataset_name: str = "celeba"
    dataroot: str = "celeba/"
    batch_size: int = 256
    image_size: int = 64
    workers: int = 4

    # Trainer
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    name: str = "debug_run"
    len_epoch: int = 1000
    log_step: int = 100
    save_step: int = 100
    num_epochs: int = 30
    ngpu: int = 1
    fid_path_to_feats: str = (
        "/home/hdilab04/arturgimranov/boutique_generative/fid_celeba_feats.pt"
    )

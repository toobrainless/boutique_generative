import os
from itertools import repeat
from pathlib import Path

from matplotlib import animation
from matplotlib import pyplot as plt
from torch import nn

ROOT_DIR: Path = Path(__file__).absolute().parent


def create_gif(img_list, gif_path):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(img, animated=True)] for img in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )
    ani.save(gif_path, writer="imagemagick", fps=10)
    plt.close()


def chdir(config):
    base_dir = ROOT_DIR / "runs" / config.name

    if not base_dir.exists():
        base_dir.mkdir(parents=True)
    else:
        version = 1
        while Path(f"{str(base_dir)}_{version}").exists():
            version += 1
        base_dir = Path(f"{base_dir}_{version}")
        base_dir.mkdir()

    (base_dir / "images").mkdir()
    (base_dir / "checkpoints").mkdir()

    os.chdir(base_dir)


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

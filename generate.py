import argparse

import torch
from torchvision.utils import save_image

from config import Config
from models import Generator


def main(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    config = Config()
    netG = Generator(config.ngpu, config.nz, config.ngf, config.nc).to(config.device)
    netG.load_state_dict(checkpoint["generator"])

    noise = torch.randn(64, config.nz, 1, 1, device=config.device)

    fake = netG(noise).detach().cpu()
    save_image(fake, "generated_images.png", normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "-cp", type=str, default="checkpoint.pth")
    args = parser.parse_args()
    main(args.checkpoint_path)

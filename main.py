from config import Config
from models import Discriminator, Generator
from trainer import train
from utils import ROOT_DIR, initialize_weights, load_dataset


def main(config):
    _, dataloader = load_dataset(
        config.dataset_name,
        config.image_size,
        ROOT_DIR / config.dataroot,
        config.batch_size,
        config.workers,
    )

    netG = Generator(config.ngpu, config.nz, config.ngf, config.nc).to(config.device)
    netD = Discriminator(config.ngpu, config.ndf, config.nc).to(config.device)
    netG.apply(initialize_weights)
    netD.apply(initialize_weights)

    train(netD, netG, dataloader, config)


if __name__ == "__main__":
    config = Config()
    main(config)

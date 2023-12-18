from dataclasses import asdict
from pathlib import Path

import torch
from piq import FID, SSIMLoss
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import wandb
from utils import chdir, create_gif, inf_loop


class FIDMetric:
    def __init__(
        self,
        dataloader=None,
        path_to_feats=None,
        device=torch.device("cpu"),
    ):
        assert (
            dataloader is not None or path_to_feats is not None
        ), "you need to provide either dataloader or paths_to_feats"
        self.metric: FID = FID()
        if path_to_feats is None:
            self.feats = self.metric.compute_feats(dataloader, device=device)
        else:
            self.feats = torch.load(path_to_feats)
        self.device = device

    def __call__(self, fake):
        fake_features = self.metric.compute_feats(
            [{"images": (fake + 1) / 2}], device=self.device
        )
        return self.metric.compute_metric(self.feats, fake_features)


def update_discriminator(
    discriminator, real_data, fake_data, optimizerD, criterion, device
):
    b_size = real_data.size(0)
    real_label = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
    fake_label = torch.full((b_size,), 0.0, dtype=torch.float, device=device)

    discriminator.zero_grad()
    # Real data
    output_real = discriminator(real_data).view(-1)
    errD_real = criterion(output_real, real_label)
    errD_real.backward()

    # Fake data
    output_fake = discriminator(fake_data.detach()).view(-1)
    errD_fake = criterion(output_fake, fake_label)
    errD_fake.backward()

    optimizerD.step()

    return (
        errD_real.item() + errD_fake.item(),
        output_real.mean().item(),
        output_fake.mean().item(),
    )


def update_generator(generator, discriminator, noise, optimizerG, criterion, device):
    generator.zero_grad()
    fake_data = generator(noise)
    label = torch.full((noise.size(0),), 1.0, dtype=torch.float, device=device)
    output = discriminator(fake_data).view(-1)
    errG = criterion(output, label)
    errG.backward()
    optimizerG.step()

    return errG.item(), output.mean().item()


def log_generated_images(generator, fixed_noise, epoch, device):
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        save_image(
            fake,
            Path("images") / f"generated_images_epoch_{epoch}.png",
            normalize=True,
        )

        wandb.log({"Generated Images": [wandb.Image(fake, caption=f"Epoch {epoch}")]})
        img = make_grid(fake, padding=2, normalize=True)

    return fake, img.permute((1, 2, 0)).numpy()


def train(discriminator, generator, dataloader, config):
    wandb.init(project="gan_project", name=config.name, config=asdict(config))
    chdir(config)

    criterion = torch.nn.BCELoss()
    optimizerD = torch.optim.Adam(
        discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
    )
    optimizerG = torch.optim.Adam(
        generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999)
    )
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=config.device)

    img_list = []

    fid_metric = FIDMetric(path_to_feats=config.fid_path_to_feats, device=config.device)
    ssim_metric = SSIMLoss()

    dataloader = inf_loop(dataloader)
    step = 0

    for epoch in range(config.num_epochs):
        data_iter = iter(dataloader)
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        for i in tqdm(
            range(config.len_epoch),
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            unit="batch",
        ):
            step += 1
            real_data = next(data_iter)[0].to(config.device)
            noise = torch.randn(
                real_data.size(0), config.nz, 1, 1, device=config.device
            )

            # Update Discriminator
            errD, D_x, D_G_z1 = update_discriminator(
                discriminator,
                real_data,
                generator(noise),
                optimizerD,
                criterion,
                config.device,
            )
            epoch_loss_D += errD

            # Update Generator
            errG, D_G_z2 = update_generator(
                generator,
                discriminator,
                noise,
                optimizerG,
                criterion,
                config.device,
            )
            epoch_loss_G += errG

            if i % config.log_step == 0:
                wandb.log(
                    {
                        "Loss_D": epoch_loss_D / config.log_step,
                        "Loss_G": epoch_loss_G / config.log_step,
                        "D(x)": D_x,
                        "D(G(z))1": D_G_z1,
                        "D(G(z))2": D_G_z2,
                    },
                    step=step,
                )
                epoch_loss_D = 0.0
                epoch_loss_G = 0.0
                fake, img = log_generated_images(
                    generator, fixed_noise, epoch, config.device
                )
                img_list.append(img)

        # Log generated images after each epoch

        fake = (fake.cpu() + 1) / 2
        wandb.log(
            {
                "FID": fid_metric(fake),
                "SSIM": ssim_metric(
                    (fake + 1) / 2, (real_data.cpu()[: fake.shape[0]] + 1) / 2
                ),
            },
            step=step,
        )

        # Save models every save_step epochs
        if epoch % config.save_step == 0 or epoch == config.num_epochs - 1:
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                },
                Path("checkpoints") / f"epoch_{epoch}.pth",
            )

    create_gif(img_list, "training_progress.gif")
    wandb.finish()

import torch
from torchvision import datasets, transforms


def load_dataset(name, image_size, dataroot, batch_size, workers):
    if name == "celeba":
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.ImageFolder(root=dataroot, transform=transform)
    else:
        raise NotImplementedError("Dataset not supported")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    return dataset, dataloader

import csv
import os
from typing import Optional, Callable, Any

import torch
import torchvision


from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR


image_size = 64
max_images = 2 ** 14


class ChexpertSmallV1(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.resizeTransform = torchvision.transforms.Resize((image_size, image_size))
        self.toPIL = torchvision.transforms.ToPILImage()

        with open(os.path.join(root, "CheXpert-v1.0-small", "train.csv")) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            self.labels = next(reader)
            self.imageLabels = []
            for row in reader:
                self.imageLabels.append(row[0])

    def __len__(self) -> int:
        return min(max_images, len(self.imageLabels))

    def __getitem__(self, index: int) -> Any:
        img_path = os.path.join(self.root, self.imageLabels[index])
        image = torchvision.io.read_image(img_path)
        image = image.repeat(3, 1, 1)
        image = self.toPIL(image)
        image = self.resizeTransform(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, torch.ones(1)


def main(args):
    batch_size = args.batch_size

    learning_rate = args.learning_rate

    device = torch.device("cuda")

    chexpert = ChexpertSmallV1("", transform=TransformsSimCLR(image_size))
    data_loader = torch.utils.data.DataLoader(chexpert, batch_size=batch_size, shuffle=True)

    encoder = get_resnet("resnet18", args.pretrained)
    n_features = encoder.fc.in_features

    model = SimCLR(encoder, args.projection_dim, n_features)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    criterion = NT_Xent(args.batch_size, args.temperature, 1)

    for epoch in range(args.epochs):
        loss_epoch = 0.0

        for step, ((x_i, x_j), _) in enumerate(data_loader):
            optimizer.zero_grad()

            x_i = x_i.to(device)
            x_j = x_j.to(device)

            h_i, h_j, z_i, z_j = model(x_i, x_j)

            loss = criterion(z_i, z_j)

            optimizer.step()

            loss_epoch += loss.item()
        
        print("Epoch {}: loss {}".format(epoch, loss_epoch))

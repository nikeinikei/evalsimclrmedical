from argparse import ArgumentParser

import torch
import torchvision
from simclr.modules import get_resnet

from train import ChexpertSmallV1


class Classifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, num_features: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes), torch.nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.mlp(z)


def main(args):
    device = torch.device("cuda")

    dataset = ChexpertSmallV1(args.chexpert_path, transform=torchvision.transforms.ToTensor(), max_images=2**14)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    encoder = get_resnet("resnet18", False)
    num_features = encoder.fc.in_features
    encoder.fc = torch.nn.Identity()

    weights = torch.load("model.pt")
    encoder.load_state_dict(weights)

    model = Classifier(encoder, num_features=num_features, num_classes=18)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    for epoch in range(args.epochs):
        loss_epoch = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat = model(x)

            loss = torch.nn.functional.mse_loss(y, y_hat)
            loss.backward()

            optimizer.step()

            loss_epoch += loss.sum().item()

        print("Epoch {}: loss {}".format(epoch, loss_epoch))

    torch.save(model.state_dict(), "model_supervised.pt")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--chexpert_path", action="store", type=str, default="")
    parser.add_argument("--batch_size", action="store", type=int, default=256)
    parser.add_argument("--learning_rate", action="store", type=float, default=1e-4)
    parser.add_argument("--epochs", action="store", type=int, default=100)
    args = parser.parse_args()
    main(args)

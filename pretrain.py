from argparse import ArgumentParser

import torch
import wandb

from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from dataset import ChexpertSmallV1, ChexpertSmallV1MICLE, image_size


def main(args):
    if args.no_dry_run:
        wandb.init(project="seminar-pretraining", config=args)

    batch_size = args.batch_size

    learning_rate = args.learning_rate

    device = torch.device("cuda")

    max_images = 2 ** 14
    if args.micle:
        dataset = ChexpertSmallV1MICLE(args.chexpert_path, max_images = max_images)
    else:
        dataset = ChexpertSmallV1("", max_images = max_images, transform=TransformsSimCLR(image_size))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_path = args.output

    encoder = get_resnet("resnet18", args.pretrained)
    if args.restore:
        weights = torch.load(model_path)
        encoder.load_state_dict(weights)
    n_features = encoder.fc.in_features

    model = SimCLR(encoder, args.projection_dim, n_features)
    model.to(device)

    if args.no_dry_run:
        wandb.watch(model)

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
            loss.backward()

            optimizer.step()

            loss_epoch += loss.item()

        print("Epoch {}: loss {}".format(epoch, loss_epoch))

        if args.no_dry_run:
            wandb.log({"loss": loss_epoch})

    torch.save(model.encoder.state_dict(), model_path)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--chexpert_path", action="store", type=str, default="")
    parser.add_argument("--batch_size", action="store", type=int, default=256)
    parser.add_argument("--learning_rate", action="store", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--temperature", action="store", type=float, default=0.5)
    parser.add_argument("--epochs", action="store", type=int, default=100)
    parser.add_argument("--projection_dim", action="store", type=int, default=64)
    parser.add_argument("--micle", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--no_dry_run", action="store_true")
    parser.add_argument("output", action="store", type=str)
    args = parser.parse_args()
    main(args)

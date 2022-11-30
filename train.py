import torch
import wandb

from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from dataset import ChexpertSmallV1, image_size


def main(args):
    wandb.init(project="seminar", config=args)

    batch_size = args.batch_size

    learning_rate = args.learning_rate

    device = torch.device("cuda")

    chexpert = ChexpertSmallV1("", max_images = 2 ** 14, transform=TransformsSimCLR(image_size))
    data_loader = torch.utils.data.DataLoader(chexpert, batch_size=batch_size, shuffle=False)

    model_path = "model.pt"

    encoder = get_resnet("resnet18", args.pretrained)
    if args.restore:
        weights = torch.load(model_path)
        encoder.load_state_dict(weights)
    n_features = encoder.fc.in_features

    model = SimCLR(encoder, args.projection_dim, n_features)
    model.to(device)

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
        wandb.log({
            "loss": loss_epoch
        })

    torch.save(model.encoder.state_dict(), model_path)

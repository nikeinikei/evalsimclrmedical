from argparse import ArgumentParser

import torch
import torchvision
from simclr.modules import get_resnet

from train_supervised import Classifier
from dataset import ChexpertSmallV1


def main(args):
    device = torch.device("cuda")

    weights = torch.load(args.model)

    encoder = get_resnet("resnet18", False)
    num_features = encoder.fc.in_features
    encoder.fc = torch.nn.Identity()
    model = Classifier(encoder, num_features, 18)
    model.load_state_dict(weights)
    model.to(device)

    dataset = ChexpertSmallV1("", max_images=2**14, validation=True, transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        correct_predictions_count = 0

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            y_hat = torch.round(y_hat)

            correct_predictions_count += torch.sum(y_hat == y).sum()

        total_predictions = len(dataset) * 18

        accuracy = correct_predictions_count / total_predictions
        print("[{}] accuracy: {}".format(args.label, accuracy))


if __name__=="__main__":
    parser = ArgumentParser("evaluation")
    parser.add_argument("--batch_size", action="store", type=int, default=256)
    parser.add_argument("--label", action="store", type=str, default="unlabeled")
    parser.add_argument("model", action="store", type=str)
    args = parser.parse_args()
    main(args)

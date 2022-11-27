from argparse import ArgumentParser
from train import main


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--chexpert_path", action="store", type=str, default="")
    parser.add_argument("--batch_size", action="store", type=int, default=256)
    parser.add_argument("--learning_rate", action="store", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--temperature", action="store", type=float, default=0.5)
    parser.add_argument("--epochs", action="store", type=int, default=10)
    parser.add_argument("--projection_dim", action="store", type=int, default=64)
    args = parser.parse_args()
    main(args)

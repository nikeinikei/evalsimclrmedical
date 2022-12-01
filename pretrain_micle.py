from argparse import ArgumentParser

import torchvision


class TransformMICLE:
    def __init__(self, size):


        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x, y):
        return self.train_transform(x), self.train_transform(y)


class LFWPairsMICLE(torchvision.datasets.VisionDataset):
    def __init__(self, root):
        self.lfw_pairs = torchvision.datasets.LFWPairs(root, download=True)

        size = (250, 250)
        color_jitter = torchvision.transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.8
        )
        self.random_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lfw_pairs)

    def __getitem__(self, index):
        img1, img2, t = self.lfw_pairs[index]

        # img1, img2 = self.random_transform(img1), self.random_transform(img2)
        print(img1)
        return img1, img2, t


def main(args):
    dataset = LFWPairsMICLE("/home/niki/datasets")
    print(dataset[0])


if __name__=="__main__":
    parser = ArgumentParser("micle training")
    parser.add_argument("--epochs", action="store", type=int, default=100)
    parser.add_argument("--no_dry_run", action="store_true")
    args = parser.parse_args()
    main(args)

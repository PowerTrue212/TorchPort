import torch

from packnet.factory import make_net as build


def main():
    model = build()
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    print(model(value).shape)


if __name__ == "__main__":
    main()

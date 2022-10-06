import argparse
import torch

def parse_args_MIWAE():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")

    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")

    parser.add_argument("--latent_dim", type=int, default=50, help="Latent variable dimension for the columns")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch-size during training")

    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


# security check
# if __name__ == "__main__":
#     args = parse_args()
#     print(args.coordinate_descent)
#
#     if not args.coordinate_descent:
#         print('blabla')

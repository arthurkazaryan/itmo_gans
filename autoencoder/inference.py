from pathlib import Path

import numpy as np
import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from model import ConvVariationalAutoencoder
from train import DEVICE, CHANNELS, get_dataset


def interpolate(autoencoder, x_1, x_2, path_to_save, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path_to_save)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to a PyTorch VAE model")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save an image")
    parser.add_argument("--start_num", required=True, type=int, help="Number to start from")
    parser.add_argument("--end_num", required=True, type=int, help="Number to finish")
    parser.add_argument("--latent_dims", required=True, type=int, default=10, help="Size of a latent dimension")

    args = parser.parse_args()

    assert Path(args.model_path).is_file(), "Model was not found."

    vae = ConvVariationalAutoencoder(CHANNELS, args.latent_dims)
    vae.load_state_dict(torch.load(args.model_path))
    vae.eval().to(DEVICE)

    data = get_dataset(batch_size=512)

    xs, ys = next(iter(data))
    x_start = xs[ys == args.start_num][0].to(DEVICE)
    x_end = xs[ys == args.end_num][0].to(DEVICE)
    interpolate(vae, x_start, x_end, n=20, path_to_save=args.save_path)

import torch
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm

from model import ConvVariationalAutoencoder


def get_dataset(batch_size: int):
    dataset = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=batch_size,
        shuffle=True
    )
    return dataset


def train(autoencoder, dataset, epochs):
    opt = torch.optim.Adam(autoencoder.parameters())
    for _ in tqdm(range(epochs)):
        for x, y in dataset:
            x = x.to(DEVICE)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()

    return autoencoder


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNELS = 1

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--epochs", required=True, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", required=True, type=int, help="Batch size")
    parser.add_argument("--latent_dims", required=True, type=int, default=10, help="Size of a latent dimension")
    parser.add_argument("--save_path", required=False, type=str, help="Path to save model", default='./vae_model.pt')

    args = parser.parse_args()

    data = get_dataset(args.batch_size)
    vae = ConvVariationalAutoencoder(CHANNELS, args.latent_dims).to(DEVICE)
    vae = train(vae, data, args.epochs)
    if args.save_path:
        torch.save(vae.state_dict(), args.save_path)

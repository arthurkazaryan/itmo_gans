import torch
import torch.nn as nn
import torch.nn.functional as functional

torch.manual_seed(0)


class ConvVariationalEncoder(nn.Module):
    def __init__(self, channels: int, latent_dims: int):
        super(ConvVariationalEncoder, self).__init__()
        self.conv_1 = nn.Conv2d(channels, 8, 3)
        self.conv_2 = nn.Conv2d(8, 16, 3)

        self.linear_1 = nn.Linear(16 * 24 * 24, latent_dims)
        self.linear_2 = nn.Linear(16 * 24 * 24, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = functional.relu(self.conv_1(x))
        x = functional.relu(self.conv_2(x))
        x = x.view(-1, 16 * 24 * 24)

        mu = self.linear_1(x)
        sigma = torch.exp(self.linear_2(x))

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class ConvDecoder(nn.Module):
    def __init__(self, channels, latent_dims):
        super(ConvDecoder, self).__init__()
        self.linear_1 = nn.Linear(latent_dims, 16 * 24 * 24)

        self.conv_t_1 = nn.ConvTranspose2d(16, 8, 3)
        self.conv_t_2 = nn.ConvTranspose2d(8, channels, 3)

    def forward(self, z):
        z = functional.relu(self.linear_1(z))
        z = z.view(-1, 16, 24, 24)
        z = functional.relu(self.conv_t_1(z))
        z = torch.sigmoid(self.conv_t_2(z))
        return z


class ConvVariationalAutoencoder(nn.Module):
    def __init__(self, channels, latent_dims):
        super(ConvVariationalAutoencoder, self).__init__()
        self.encoder = ConvVariationalEncoder(channels, latent_dims)
        self.decoder = ConvDecoder(channels, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

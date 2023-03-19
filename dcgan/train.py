from pathlib import Path
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from argparse import ArgumentParser

from discriminator import Discriminator
from generator import CSPGenerator, ResNetGenerator, GeneratorTypes
from settings import VECTOR_SIZE, LEARNING_RATE, BETA1, IMAGE_SIZE, WORKERS


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def load_dataset(data_root, batch_size):
    zip_path = Path('img_align_celeba.zip')
    dir_path = Path(data_root, 'img_align_celeba')
    if not zip_path.is_file():  # or not dir_path.is_dir():
        raise FileNotFoundError("Zip-file \"img_align_celeba.zip\" was not found.")
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)

    dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(IMAGE_SIZE),
                                   transforms.CenterCrop(IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=WORKERS)

    return dataloader


def train(dataloader, discriminator, generator, loss, optimizer_d, optimizer_g, epochs):

    img_list = []
    g_losses = []
    d_losses = []
    iters = 0

    real_label = 1.
    fake_label = 0.
    fixed_noise = torch.randn(64, VECTOR_SIZE, 1, 1, device=DEVICE)

    print("Starting Training Loop...")
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            net_d.zero_grad()
            # Format batch
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D
            output = net_d(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = loss(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, VECTOR_SIZE, 1, 1, device=DEVICE)
            # Generate fake image batch with G
            fake = net_g(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = net_d(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = loss(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = net_d(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_g.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            g_losses.append(errG.item())
            d_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    return discriminator, generator


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str, help="Path to a dataset root")
    parser.add_argument("--generator", required=True, type=str, help="Type of a generator to train")
    parser.add_argument("--epochs", required=True, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", required=True, type=int, help="Batch size")
    parser.add_argument("--save_path", required=False, type=str, help="Path to save model")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    assert data_root.is_dir(), "Dataset was not found"
    assert GeneratorTypes.has_value(args.generator), "Generator is unavailable"

    data_loader = load_dataset(data_root, args.epochs)

    if args.generator == GeneratorTypes.csp_gan.value:
        net_g = CSPGenerator().to(DEVICE)
    else:
        net_g = ResNetGenerator().to(DEVICE)
    net_d = Discriminator().to(DEVICE)

    net_g.apply(weights_init)
    net_d.apply(weights_init)

    criterion = nn.BCELoss()

    opt_d = optim.Adam(net_d.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    opt_g = optim.Adam(net_g.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    net_d, net_g = train(
        dataloader=data_loader,
        discriminator=net_d,
        generator=net_g,
        loss=criterion,
        optimizer_d=opt_d,
        optimizer_g=opt_g,
        epochs=args.epochs
    )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(net_d.state_dict(), save_path.joinpath('discriminator.pt'))
        torch.save(net_g.state_dict(), save_path.joinpath('generator.pt'))

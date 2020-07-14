import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

from utils import print_network, DEVICE


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = std.new_tensor(torch.randn(std.size()))
    return mu + std * eps


batch_size = 64
train_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

z_dim = 32


class generator(nn.Module):
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self):
        super(generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )
        self.to(DEVICE)

    def forward(self, z):
        fc_out = self.fc(z)
        fc_out = fc_out.view(-1, 128, 7, 7)
        x = self.deconv(fc_out)
        return x


class discriminator(nn.Module):
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1 + z_dim * 2)
        )
        self.to(DEVICE)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        a = F.sigmoid(x[:, 0])
        mu = x[:, 1:1 + z_dim]
        logvar = x[:, 1 + z_dim:]
        return a, mu, logvar


class info_gan():
    def __init__(self, save_dir='./info_gan/', g_lr=1e-3, d_lr=5e-4,
                 n_epochs=10, beta=1.0, display_interval=100, ):
        os.makedirs(save_dir, exist_ok=True)
        self.G = generator()
        self.D = discriminator()
        self.data_iter = train_iter
        self.save_dir = save_dir
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.n_epochs = n_epochs
        self.beta = beta
        self.display_interval = display_interval

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.g_trainer = optim.Adam(self.G.parameters(), lr=self.g_lr, betas=[0.5, 0.99])
        self.d_trainer = optim.Adam(self.D.parameters(), lr=self.d_lr, betas=[0.5, 0.99])

        print('-' * 20 + 'network' + '-' * 20)
        print_network(self.G)
        print_network(self.D)

    def train(self):
        print('-' * 50 + '\ntrain...')
        for e in range(self.n_epochs):
            self.G.train()
            self.D.train()
            for i, (x, l) in enumerate(self.data_iter):
                bs = x.size(0)
                z = torch.randn(bs, z_dim).to(DEVICE)

                # train D
                real_x = x.to(DEVICE)
                real_score, _, _ = self.D(real_x)
                fake_x = self.G(z)
                fake_score, mu, logvar = self.D(fake_x.detach())
                fake_z = reparametrize(mu, logvar)

                loss_d = self.bce_criterion(real_score, torch.ones_like(real_score)) + \
                         self.bce_criterion(fake_score, torch.zeros_like(fake_score))
                loss_d += self.beta * self.mse_criterion(fake_z, z)

                self.D.zero_grad()
                loss_d.backward()
                self.d_trainer.step()

                # train G
                real_score, _, _ = self.D(real_x)
                fake_score, mu, logvar = self.D(fake_x)
                fake_z = reparametrize(mu, logvar)
                loss_g = self.bce_criterion(fake_score, torch.ones_like(fake_score))
                loss_g += self.beta * self.mse_criterion(fake_z, z)

                self.G.zero_grad()
                loss_g.backward()
                self.g_trainer.step()

                if (i + 1) % self.display_interval == 0:
                    tv.utils.save_image(fake_x[:16], self.save_dir + '{}_{}.png'.format(e + 1, i + 1))
                    print('[%2d/%2d] loss_d: %.3f loss_g: %.3f r_score: %.3f f_score: %.3f' % (
                        e + 1, self.n_epochs, loss_d.item(), loss_g.item(), torch.mean(real_score),
                        torch.mean(fake_score)
                    ))
            # save
            if (e + 1) % 10 == 0:
                torch.save(self.G.state_dict(), self.save_dir + 'net_g_{}.pth'.format(e + 1))
                torch.save(self.D.state_dict(), self.save_dir + 'net_d_{}.pth'.format(e + 1))

            # test
            with torch.no_grad():
                # 1: z连续插值
                batch_size = 8
                for dim in range(z_dim):
                    z = torch.randn(z_dim).repeat(batch_size, 1)
                    z[:, dim] = torch.linspace(-1.5, 1.5, batch_size)
                    fake_x = self.G(z.to(DEVICE))
                    tv.utils.save_image(fake_x, self.save_dir + 'test_e{}_d{}.png'.format(e + 1, dim))


if __name__ == '__main__':
    model = info_gan(n_epochs=20, beta=1.0, save_dir='./info_gan_t5/')
    model.train()

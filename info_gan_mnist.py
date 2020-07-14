import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

from utils import one_hot, print_network, DEVICE

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


class generator(nn.Module):
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self):
        super(generator, self).__init__()
        self.zc_dim = 62 + 12
        self.fc = nn.Sequential(
            nn.Linear(self.zc_dim, 1024),
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

    def forward(self, z, dist_code, cont_code):
        zc = torch.cat([z, cont_code, dist_code], 1)
        fc_out = self.fc(zc)
        fc_out = fc_out.view(-1, 128, 7, 7)
        x = self.deconv(fc_out)
        return x


class discriminator(nn.Module):
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(discriminator, self).__init__()
        self.len_disc = 10
        self.len_cont = 2
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
            nn.Linear(1024, 1 + self.len_disc + self.len_cont)
        )
        self.to(DEVICE)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        a = F.sigmoid(x[:, 0])
        b = x[:, 1:1 + self.len_disc]
        c = x[:, 1 + self.len_disc:]
        return a, b, c


class info_gan():
    def __init__(self, save_dir='./info_gan/',
                 supervised=True, g_lr=1e-3, d_lr=5e-4,
                 n_epochs=10, beta=1, display_interval=100, ):
        os.makedirs(save_dir, exist_ok=True)
        self.G = generator()
        self.D = discriminator()
        self.data_iter = train_iter
        self.save_dir = save_dir
        self.supervised = supervised
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.n_epochs = n_epochs
        self.beta = beta
        self.display_interval = display_interval

        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
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
                batch_size = x.size(0)
                z = torch.randn(batch_size, 62).to(DEVICE)
                if self.supervised:
                    dist = l.to(DEVICE)
                else:
                    dist = torch.randint(0, 9, (batch_size,), dtype=torch.long).to(DEVICE)
                one_hot_dist = one_hot(dist.cpu(), 10).to(DEVICE)
                cont = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, 2))).float().to(DEVICE)

                # train D
                real_x = x.to(DEVICE)
                a, b, c = self.D(real_x)
                fake_x = self.G(z, one_hot_dist, cont)
                a_, b_, c_ = self.D(fake_x.detach())

                loss_d = self.bce_criterion(a, torch.ones_like(a)) + self.bce_criterion(a_, torch.zeros_like(a_))
                loss_d += self.beta * self.ce_criterion(b_, dist)
                loss_d += self.beta * self.mse_criterion(c_, cont)

                self.D.zero_grad()
                loss_d.backward()
                self.d_trainer.step()

                # train G
                a, _, _ = self.D(real_x)
                a_, b_, c_ = self.D(fake_x)
                loss_g = self.bce_criterion(a_, torch.ones_like(a_))
                loss_g += self.beta * self.ce_criterion(b_, dist)
                loss_g += self.beta * self.mse_criterion(c_, cont)

                self.G.zero_grad()
                loss_g.backward()
                self.g_trainer.step()

                if (i + 1) % self.display_interval == 0:
                    print('[%2d/%2d] loss_d: %.3f loss_g: %.3f r_score: %.3f f_score: %.3f' % (
                        e + 1, self.n_epochs, loss_d.item(), loss_g.item(), torch.mean(a), torch.mean(a_)
                    ))
            # save
            if (e + 1) % 10 == 0:
                torch.save(self.G.state_dict(), self.save_dir + 'net_g_{}.pth'.format(e + 1))
                torch.save(self.D.state_dict(), self.save_dir + 'net_d_{}.pth'.format(e + 1))

            # test
            with torch.no_grad():
                # 1：随机生成
                self.G.eval()
                self.D.eval()
                batch_size = 16
                z = torch.randn(batch_size, 62).to(DEVICE)
                dist = torch.randint(0, 9, (batch_size,), dtype=torch.long).to(DEVICE)
                one_hot_dist = one_hot(dist.cpu(), 10).to(DEVICE)
                cont = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, 2))).float().to(DEVICE)
                fake_x = self.G(z, one_hot_dist, cont)
                tv.utils.save_image(fake_x, self.save_dir + 't1_{}.png'.format(e + 1))
                # 2: 只改变离散code
                batch_size = 10
                z = torch.randn(1, 62).repeat(batch_size, 1).to(DEVICE)
                one_hot_dist = one_hot(torch.range(0, 9, dtype=torch.long), 10).to(DEVICE)
                cont = torch.from_numpy(np.random.uniform(-1, 1, size=(1, 2))).repeat(batch_size, 1).float().to(DEVICE)
                fake_x = self.G(z, one_hot_dist, cont)
                tv.utils.save_image(fake_x, self.save_dir + 't2_{}.png'.format(e + 1))
                # 3: 连续code 01
                batch_size = 8
                z = torch.randn(1, 62).repeat(batch_size, 1).to(DEVICE)
                dist = torch.randint(0, 9, (1, 1), dtype=torch.long)
                one_hot_dist = one_hot(dist, 10).repeat(batch_size, 1).to(DEVICE)
                cont01 = torch.linspace(-1, 1, batch_size).view(batch_size, 1).to(DEVICE)
                cont02 = torch.from_numpy(np.random.uniform(-1, 1, size=(1, 1))).repeat(batch_size, 1).float().to(
                    DEVICE)
                cont = torch.cat([cont01, cont02], 1)
                fake_x = self.G(z, one_hot_dist, cont)
                tv.utils.save_image(fake_x, self.save_dir + 't3_{}.png'.format(e + 1))

                # 4: 连续code 02
                batch_size = 8
                z = torch.randn(batch_size, 62).to(DEVICE)
                dist = torch.randint(0, 9, (1, 1), dtype=torch.long)
                one_hot_dist = one_hot(dist, 10).repeat(batch_size, 1).to(DEVICE)
                cont02 = torch.linspace(-1, 1, batch_size).view(batch_size, 1).to(DEVICE)
                cont01 = torch.from_numpy(np.random.uniform(-1, 1, size=(1, 1))).repeat(batch_size, 1).float().to(
                    DEVICE)
                cont = torch.cat([cont01, cont02], 1)
                fake_x = self.G(z, one_hot_dist, cont)
                tv.utils.save_image(fake_x, self.save_dir + 't4_{}.png'.format(e + 1))


if __name__ == '__main__':
    model = info_gan(n_epochs=20, supervised=True)
    model.train()

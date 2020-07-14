import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

from utils import *

_transformer = tv.transforms.Compose([
    tv.transforms.Resize([64, 64]),
    tv.transforms.ToTensor()
])
chairs_3d_iter = torch.utils.data.DataLoader(
    dataset=single_class_image_folder('../../Datasets/rendered_chairs/', transform=_transformer),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=4,
)

len_z = 128
len_disc_1 = 20
len_disc_2 = 20
len_disc_3 = 20
len_cont = 1


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.zc_dim = len_z + len_disc_1 + len_disc_2 + len_disc_3 + len_cont
        self.fc = nn.Sequential(
            nn.Linear(self.zc_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.ConvTranspose2d(256, 256, 4, 1, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.ConvTranspose2d(256, 256, 4, 1, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        self.to(DEVICE)

    def forward(self, z, dist_code_1, dist_code_2, dist_code_3, cont_code):
        zc = torch.cat([z, dist_code_1, dist_code_2, dist_code_3, cont_code], 1)
        fc_out = self.fc(zc)
        fc_out = fc_out.view(-1, 256, 8, 8)
        x = self.deconv(fc_out)
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.Conv2d(256, 256, 4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.Conv2d(256, 256, 4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1 + len_disc_1 + len_disc_2 + len_disc_3 + len_cont)
        )
        self.to(DEVICE)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 256 * 8 * 8)
        x = self.fc(x)
        a = F.sigmoid(x[:, 0])
        d1 = x[:, 1:1 + len_disc_1]
        d2 = x[:, 1 + len_disc_1: 1 + len_disc_1 + len_disc_2]
        d3 = x[:, 1 + len_disc_1 + len_disc_2:1 + len_disc_1 + len_disc_2 + len_disc_3]
        c = x[:, 1 + len_disc_1 + len_disc_2 + len_disc_3:]
        return a, d1, d2, d3, c


class info_gan():
    # Rotation: disc_beta=1.0, cont_beta=10.0
    # Width: disc_beta=2.0, cont_beta=0.05
    def __init__(self, save_dir='./info_gan_3dchairs/', g_lr=1e-3, d_lr=2e-4,
                 n_epochs=10, disc_beta=1.0, cont_beta=10.0, display_interval=100):
        os.makedirs(save_dir, exist_ok=True)
        self.G = generator()
        self.D = discriminator()
        self.data_iter = chairs_3d_iter
        self.save_dir = save_dir
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.n_epochs = n_epochs
        self.disc_beta = disc_beta
        self.cont_beta = cont_beta
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
            for i, x in enumerate(self.data_iter):
                batch_size = x.size(0)
                z = torch.randn(batch_size, len_z).to(DEVICE)
                dist_1 = torch.randint(0, len_disc_1 - 1, (batch_size,), dtype=torch.long).to(DEVICE)
                one_hot_dist_1 = one_hot(dist_1.cpu(), len_disc_1).to(DEVICE)
                dist_2 = torch.randint(0, len_disc_2 - 1, (batch_size,), dtype=torch.long).to(DEVICE)
                one_hot_dist_2 = one_hot(dist_2.cpu(), len_disc_2).to(DEVICE)
                dist_3 = torch.randint(0, len_disc_3 - 1, (batch_size,), dtype=torch.long).to(DEVICE)
                one_hot_dist_3 = one_hot(dist_3.cpu(), len_disc_3).to(DEVICE)
                cont = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, len_cont))).float().to(DEVICE)

                # train D
                real_x = x.to(DEVICE)
                a, d1, d2, d3, c = self.D(real_x)
                fake_x = self.G(z, one_hot_dist_1, one_hot_dist_2, one_hot_dist_3, cont)
                a_, d1_, d2_, d3_, c_ = self.D(fake_x.detach())

                loss_d = self.bce_criterion(a, torch.ones_like(a)) + self.bce_criterion(a_, torch.zeros_like(a_))
                loss_d += self.disc_beta * self.ce_criterion(d1_, dist_1)
                loss_d += self.disc_beta * self.ce_criterion(d2_, dist_2)
                loss_d += self.disc_beta * self.ce_criterion(d3_, dist_3)
                loss_d += self.cont_beta * self.mse_criterion(c_, cont)

                self.D.zero_grad()
                loss_d.backward()
                self.d_trainer.step()

                # train G
                a, d1, d2, d3, c = self.D(real_x)
                a_, d1_, d2_, d3_, c_ = self.D(fake_x)
                loss_g = self.bce_criterion(a_, torch.ones_like(a_))
                loss_g += self.disc_beta * self.ce_criterion(d1_, dist_1)
                loss_g += self.disc_beta * self.ce_criterion(d2_, dist_2)
                loss_g += self.disc_beta * self.ce_criterion(d3_, dist_3)
                loss_g += self.cont_beta * self.mse_criterion(c_, cont)

                self.G.zero_grad()
                loss_g.backward()
                self.g_trainer.step()

                if (i + 1) % self.display_interval == 0:
                    tv.utils.save_image(fake_x[:64], self.save_dir + '{}_{}.png'.format(e + 1, i + 1))
                    print('[%2d/%2d] loss_d: %.3f loss_g: %.3f r_score: %.3f f_score: %.3f' % (
                        e + 1, self.n_epochs, loss_d.item(), loss_g.item(), torch.mean(a), torch.mean(a_)
                    ))
            # save
            if (e + 1) % 10 == 0:
                torch.save(self.G.state_dict(), self.save_dir + 'net_g_{}.pth'.format(e + 1))
                torch.save(self.D.state_dict(), self.save_dir + 'net_d_{}.pth'.format(e + 1))

            # test
            with torch.no_grad():
                self.G.eval()
                # 1: 只改变离散code_1
                batch_size = len_disc_1
                z = torch.randn(1, len_z).repeat(batch_size, 1).to(DEVICE)
                one_hot_dist_1 = one_hot(torch.range(0, len_disc_1 - 1, dtype=torch.long), len_disc_1).to(DEVICE)
                dist_2 = torch.randint(0, len_disc_2 - 1, (1, 1), dtype=torch.long)
                one_hot_dist_2 = one_hot(dist_2, len_disc_2).repeat(batch_size, 1).to(DEVICE)
                dist_3 = torch.randint(0, len_disc_1 - 1, (1, 1), dtype=torch.long)
                one_hot_dist_3 = one_hot(dist_3.cpu(), len_disc_3).repeat(batch_size, 1).to(DEVICE)
                cont = torch.from_numpy(np.random.uniform(-1, 1, size=(1, len_cont))).repeat(batch_size, 1).float().to(
                    DEVICE)
                fake_x = self.G(z, one_hot_dist_1, one_hot_dist_2, one_hot_dist_3, cont)
                tv.utils.save_image(fake_x, self.save_dir + 't1_{}.png'.format(e + 1))


if __name__ == '__main__':
    model = info_gan(n_epochs=20)
    model.train()

from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
batch_size = 100
lr = 1e-4
latent_size = 100
num_epochs = 100
#path
save_image_dir = "save_image"
save_model_dir = "save_model"
os.makedirs(save_image_dir,exist_ok=True)
os.makedirs(save_model_dir,exist_ok=True)
writer = SummaryWriter()
# model
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.output_bias = nn.Parameter(torch.zeros(1, 28, 28), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 256, 4, stride=1, bias=False), # 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False), #10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),#13
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),#28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False), #28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 1, stride=1, bias=False)#28
        )
    def forward(self, input):
        output = self.main(input)
        output = torch.sigmoid(output + self.output_bias)
        return output
class Encoder(nn.Module):
    def __init__(self, latent_size, noise=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, bias=False), # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, bias=False),# 11
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, bias=False), #11-5/2+1 = 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.main2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=1, bias=False), #1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.main3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=1, bias=False), #1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.main4 = nn.Sequential(
            nn.Conv2d(512, self.latent_size, 1, stride=1, bias=True) #1
        )
    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        return output, x3.view(batch_size, -1), x2.view(batch_size, -1), x1.view(batch_size, -1)
class Discriminator(nn.Module):
    def __init__(self, latent_size, dropout, output_size=10):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.infer_x = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, bias=True), # 24
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(32, 64, 4, stride=2, bias=False), #24-4/2 + 1 = 11
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(64, 128, 5, stride=2, bias=False), #11-5/2+1 = 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(128, 256, 4, stride=1, bias=False), #1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(256, 512, 1, stride=1, bias=False), #1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)

        )
        self.infer_z = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )
        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )
        self.final = nn.Conv2d(1024, self.output_size, 1, stride=1, bias=True)
    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        if self.output_size == 1:
            output = torch.sigmoid(output)
        return output.squeeze(), output_features.view(x.size()[0], -1)
#model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)
def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))
def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))
#function
# transform = transforms.Compose([transforms.ToTensor()])
# # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5),std = (0.5))])
# train_set = datasets.MNIST(root="E:/data",train=True,transform=transform)
# print(train_set.class_to_idx)
# train_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size = batch_size,
#     shuffle=True
# )
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root="/home/hugogao88/data",train=True,transform=transform)
label = [x[1] for x in train_set]
idx = 0
train_filter = np.where([bool(label[i] != idx) for i in range(len(label))])
train_filter = np.array(train_filter,dtype=int)
train_filter = train_filter.squeeze()
train = []
for i in range(len(train_filter)):
  train.append(train_set[train_filter[i]])

train_loader_filter = torch.utils.data.DataLoader(
    train,
    batch_size = batch_size,
    shuffle=True
)

img,label = next(iter(train_loader_filter))
print(label)
netE = Encoder(latent_size, True)
netG = Generator(latent_size)
netD = Discriminator(latent_size, 0.2, 1)
netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)
optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()
for epoch in range(num_epochs):
    i = 0
    for (data, target) in train_loader_filter:
        real_label = torch.ones(batch_size)
        fake_label = torch.zeros(batch_size)
        noise1 = torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)
        noise2 = torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)
        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(data)
        if data.size()[0] != batch_size:
            continue
        d_real = data
        z_fake = torch.randn(batch_size, latent_size, 1, 1)
        d_fake = netG(z_fake)
        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)
        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(torch.randn(batch_size, latent_size))
        output_z = mu + epsilon * sigma
        output_real, real_feature = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, fake_feature = netD(d_fake + noise2, z_fake)
        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label) + 0.001 * torch.norm(real_feature - fake_feature, p=1)
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label) 
        # + 0.005 * torch.norm(d_real - netG(output_z.view(batch_size,latent_size,1,1)), p=1)
        if loss_g.item() < 50:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()
            output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
            output_fake, _ = netD(d_fake + noise2, z_fake)
            loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label) + 0.001 * torch.norm(real_feature - fake_feature, p=1)
            loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label) 
            # + 0.005 * torch.norm(d_real - netG(output_z.view(batch_size,latent_size,1,1)), p=1)

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()
        batches_done = len(train_loader_filter) * epoch + i
        if i % 1 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), "G loss :", loss_g.item(),
                  "D(x) :", output_real.mean().item(), "D(G(x)) :", output_fake.mean().item(),"Feature loss :",torch.norm(real_feature - fake_feature,p=1).item(),"Reconstruction loss :",torch.norm(d_real - netG(output_z.view(batch_size,latent_size,1,1)),p=1).item())
            writer.add_scalar("D loss",loss_d.item(),batches_done)
            writer.add_scalar("G loss/ add reconstruction loss",loss_g.item(),batches_done)
            writer.add_scalar("feature loss", torch.norm(real_feature - fake_feature, p=1).item(), batches_done)
            writer.add_scalar("reconstruction loss", torch.norm(d_real - netG(output_z.view(batch_size,latent_size,1,1)), p=1).item(), batches_done)
        if i % 10 == 0:
            vutils.save_image(d_fake.data[:16, ], './%s/fake_%d.png' % (save_image_dir,batches_done))
            vutils.save_image(d_real.data[:16, ], './%s/real.png'% (save_image_dir))
        i += 1
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), './%s/netG_epoch_%d.pth' % (save_model_dir, epoch))
        torch.save(netE.state_dict(), './%s/netE_epoch_%d.pth' % (save_model_dir, epoch))
        torch.save(netD.state_dict(), './%s/netD_epoch_%d.pth' % (save_model_dir, epoch))
        vutils.save_image(d_fake.data[:16, ], './%s/fake_%d.png' % (save_image_dir, epoch))
writer.close()

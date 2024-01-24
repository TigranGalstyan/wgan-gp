import os
import sys
sys.path.append(os.getcwd())

import time
import tflib as lib
import tflib.save_images
# import tflib.mnist
import tflib.cifar10
import tflib.plot
# import tflib.inception_score

import numpy as np

import torch
torch.manual_seed(123)
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchmetrics.image.fid import FrechetInceptionDistance

from pytorch_fid import fid_score, inception

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'wgan-gp'  # Valid options are dcgan, wgan, or wgan-gp
DIM = 128  # This overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64 * 4  # Batch size
ITERS = 800000  # How many generator iterations to train for
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (3*32*32)

riperm_l = 2.0
log_folder = f'./tmp/cifar10_riperm_{riperm_l}'
gpu = 1

fid = FrechetInceptionDistance(normalize=True, reset_real_features=False, feature=2048)  # feature=2048
fid.cuda(device=gpu)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output


class InverseGenerator(nn.Module):
    def __init__(self):
        super(InverseGenerator, self).__init__()
        preprocess = nn.Sequential(
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
            nn.Linear(4 * 4 * 4 * DIM, 128),
        )

        block1 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(2 * DIM),
            nn.Conv2d(2 * DIM, 4 * DIM, 2, stride=2),
        )
        block2 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(DIM),
            nn.Conv2d(DIM, 2 * DIM, 2, stride=2),
        )
        deconv_out = nn.Conv2d(3, DIM, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out

    def forward(self, input):
        output = self.deconv_out(input)
        output = self.block2(output)
        output = self.block1(output)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        output = self.preprocess(output)

        return output


def GeneratorInverseLoss(orig_latent, pred_latent):
    """
    Inverse Generator Loss
    """
    return nn.functional.mse_loss(orig_latent, pred_latent)


netG = Generator()
netD = Discriminator()
netIG = InverseGenerator()
print(netG)
print(netIG)
print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
    nerIG = netIG.cuda(gpu)

one = torch.Tensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerIG = optim.Adam(netIG.parameters(), lr=1e-4, betas=(0.5, 0.9))


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# For generating samples
def generate_image(frame, G):
    print("Generating Samples.")
    fixed_noise_128 = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    with torch.no_grad():
        noisev = autograd.Variable(fixed_noise_128)
        samples = G(noisev)
        samples = samples.view(-1, 3, 32, 32)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, f'{log_folder}/samples_{frame}.jpg')


# # For calculating inception score
def get_inception_score(G):
    print("Calculating FID scores.")
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    with torch.no_grad():
        noise_fid = autograd.Variable(fixed_noise_128)
        samples = netG(noise_fid)

    # all_samples = torch.concatenate(all_samples, dim=0)
    fid.update((samples + 1) / 2, real=False)
    score = fid.compute()
    fid.reset()
    return score


# Dataset iterator
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)


def inf_train_gen():
    while True:
        for images in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images


gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):
        _data = next(gen)
        netD.zero_grad()

        # train with real
        _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        fid.update((real_data + 1.0) / 2.0, real=True)

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)

        D_real = netD(real_data_v)
        D_real = D_real.mean(axis=0)
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        with torch.no_grad():
            noisev = autograd.Variable(noise)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean(axis=0)
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)

    predicted_latent = netIG(fake)
    inv_cost = GeneratorInverseLoss(noisev, predicted_latent)

    G_cost = - G.mean(axis=0)
    Cost = G_cost + inv_cost * riperm_l
    Cost.backward()
    optimizerG.step()
    optimizerIG.step()

    # Write logs and save samples
    lib.plot.plot(f'{log_folder}/train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot(f'{log_folder}/time', time.time() - start_time)
    lib.plot.plot(f'{log_folder}/train gen cost', G_cost.cpu().data.numpy())
    lib.plot.plot(f'{log_folder}/train inverse gen cost', inv_cost.cpu().data.numpy())
    lib.plot.plot(f'{log_folder}/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate fid every 250 iters
    # Calculate dev loss and generate samples every 100 iters
    if iteration % 250 == 0:
        current_fid = get_inception_score(netG).cpu().data
        lib.plot.plot(f'{log_folder}/fid', current_fid)

        print("Calculating Dev Cost.")
        dev_disc_costs = []
        for images in dev_gen():
            images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            imgs = torch.stack([preprocess(item) for item in images])

            # imgs = preprocess(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot(f'{log_folder}/dev disc cost', np.mean(dev_disc_costs))

        generate_image(iteration, netG)

    # Save logs every 250 iters
    if iteration % 250 == 0:
        lib.plot.flush()
    lib.plot.tick()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from channel_models import ch_SSPA

from matplotlib import pyplot as plt
from IPython.display import clear_output

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def pseudo_qcode(msgs):
     def qam_mapper(m):
        # m takes in a vector of messages
        x = np.linspace(-3,3,4)
        y = np.meshgrid(x,x)
        z = np.array(y).reshape(2,16)
        return np.array([[z[0][i],z[1][i]] for i in m])
    
     code_q = torch.from_numpy(qam_mapper(msgs)).to(device)
     code_std = torch.std(code_q)
     return code_q / code_std
    
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = torch.nn.ReLU()
    
    def forward(self, x):
        #i,j = x.shape
        noise = torch.randn_like(x).to(device)
        inputs = torch.cat([x, noise], 1)
        x = self.f(self.map1(inputs))
        x = self.f(self.map2(x))
        out = self.map3(x)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = torch.nn.ReLU()
    
    def forward(self, z, c):
        inputs = torch.cat([z, c], 1)
        x = self.f(self.map1(inputs))
        x = self.f(self.map2(x))
        x = self.map3(x)
        return x



class Generator_CNN_1D(nn.Module):
    def __init__(self, length_seq, dim):
        super(Generator_CNN_1D, self).__init__()
        assert dim>8, 'dim should be larger than 8.'
        input_size = length_seq * 2 * 2 # due to using complex domain and concatenation for conditioning.
        output_size = length_seq * 2 # size of the generated samples.

        self.length_seq = length_seq
        self.dim = dim
        self.fc1 = nn.Linear(input_size, length_seq * dim)
        self.conv1 = nn.Conv1d(dim, dim, 7, padding = 3)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding = 1)
        self.conv3 = nn.Conv1d(dim, dim//2, 3, padding = 1)
        self.conv4 = nn.Conv1d(dim//2, dim//2, 3, padding = 1)
        self.fc2 = nn.Linear(length_seq * (dim//2), length_seq * (dim//2-2) //2)
        self.fc_fin = nn.Linear(length_seq * (dim//2-2) //2, output_size)

    def forward(self, x):
        bs,i,j = x.shape
        assert i == 2, f'The input tensor should have 2 channels to represent complex values. Currently the input tensor has {i} channels.'
        assert j == self.length_seq, f'The input tensor should have the sequence length matching with the defined value, {self.length_seq}'

        noise = torch.randn_like(x).to(device)
        inputs = torch.cat([x, noise], 1)
        inputs = torch.flatten(inputs, 1)
        x1 = F.relu(self.fc1(inputs))

        x1 = x1.view(bs, self.dim, self.length_seq)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))

        x1 = torch.flatten(x1, 1)
        x1 = F.relu(self.fc2(x1))
        out = self.fc_fin(x1).view(-1, 2, self.length_seq) + x

        return out


class Generator_DCGAN_1D(nn.Module):
    def __init__(self, length_seq):
        super(Generator_DCGAN_1D, self).__init__()

        assert length_seq>16, 'sequence length should be larger than 16 to use the CNN.'
        data_size = length_seq * 2 # due to using complex domain and concatenation for conditioning.

        self.length_seq = length_seq
        dims = [1024, 512, 256, 128]
        self.dims = dims
        self.z_size = length_seq // 16

        self.main = nn.Sequential(
            nn.ConvTranspose1d(dims[0], dims[1], kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(True),
            nn.ConvTranspose1d(dims[1], dims[2], kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(True),
            nn.ConvTranspose1d(dims[2], dims[3], kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(dims[3]),
            nn.ReLU(True),
            nn.ConvTranspose1d(dims[3], 2, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

        self.emb = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(data_size, data_size)
        )
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

    def forward(self, x):
        bs,i,j = x.shape
        device = self.device
        assert i == 2, f'The input tensor should have 2 channels to represent complex values. Currently the input tensor has {i} channels.'
        assert j == self.length_seq, f'The input tensor should have the sequence length matching with the defined value, {self.length_seq}'

        inputs = torch.randn(bs ,self.dims[0], self.z_size, device=device)
        out = self.main(inputs) + self.emb(x).view(x.shape)
        return out

class Discriminator_CNN_1D(nn.Module):
    def __init__(self, length_seq, dim):
        super(Discriminator_CNN_1D, self).__init__()
        assert dim>16, 'dim should be larger than 16.'
        input_size = length_seq * 2 * 2  # due to using complex domain and concatenation for conditioning.
        output_size = 1 # size of the generated samples.

        self.conv1 = nn.Conv1d(4, dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim//2, 5, padding=1)
        self.conv3 = nn.Conv1d(dim//2, dim//4, 9, padding=1)
        self.conv4 = nn.Conv1d(dim//4, dim//8, 17, padding=1)
        hidden_size = (length_seq-2-6-14) * (dim//8) + input_size
        self.fc1 = nn.Linear( hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc3 = nn.Linear(hidden_size//4, hidden_size//8)
        self.fc_fin = nn.Linear(hidden_size//8, output_size)

        self.act = torch.nn.ReLU()

    def forward(self, z, c):
        inputs = torch.cat([z, c], 1)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(torch.cat((torch.flatten(x, 1), torch.flatten(inputs, 1)), dim=1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc_fin(x)

        return out

class Discriminator_DCGAN_1D(nn.Module):
    def __init__(self, length_seq):
        super(Discriminator_DCGAN_1D, self).__init__()

        assert length_seq>16, 'sequence length should be larger than 16.'
        dims = [128, 256, 512, 1024]

        self.main = nn.Sequential(
            nn.Conv1d(4, dims[0], kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(dims[0], dims[1], kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(dims[1]),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(dims[1], dims[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dims[2]),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(dims[2], dims[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dims[3]),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(dims[3], 1, kernel_size= length_seq // 16, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        inputs = torch.cat([z, c], 1)
        return self.main(inputs)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    fake_samples = fake_samples.to(device)
    real_samples = real_samples.to(device)
    alpha = torch.rand((real_samples.size(0),1), device= device)   #alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples

    interpolates = (alpha * real_samples + (1-alpha) * fake_samples).requires_grad_(True).to(device) #(torch.mul(alpha,real_samples) + torch.mul((1 - alpha), fake_samples)).to(device)#

    x = torch.randn_like(interpolates)
    x_std = torch.std(x)
    x_norm = x / x_std
    x_norm = x_norm#.to(device)

    d_interpolates = D(interpolates, x_norm)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

'''
def histogram_gp(G, D, batch_size, n):
    x = torch.randn((batch_size, n))
    x_std = torch.std(x)
    x_norm = x / x_std
    x_norm = x_norm.to(device)
    fake_data = G(x_norm).float().detach()
    real_data = channel(x_norm).detach()
    gradient_penalty = compute_gradient_penalty(D, real_data.data, fake_data.data)

    gradient_penalty = gradient_penalty.detach().cpu()
'''

def test_and_plot_histogram(generator, noise_std, device, num_samples=int(1e5)):
    # Generate real data
    x_real = torch.randn(num_samples, 8).to(device)
    y_real = ch_SSPA(x_real, noise_std, device)#channel_awgn(x_real)  # Additive Gaussian noise
    # Generate fake data using the generator
    with torch.no_grad():
        y_fake = generator(x_real).detach().cpu().numpy()
    # Convert real data to numpy for plotting
    y_real = y_real.detach().cpu().numpy()
    # Plot the histograms for each channel
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, tight_layout=True)
    # Channel 1
    axs[0].hist(y_real[:, 3], bins=60, alpha=0.5, label="Real")
    axs[0].hist(y_fake[:, 3], bins=60, alpha=0.5, label="Fake")
    axs[0].set_title("Channel 1")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    axs[0].grid(True)
    # Channel 2
    axs[1].hist(y_real[:, 4], bins=60, alpha=0.5, label="Real")
    axs[1].hist(y_fake[:, 4], bins=60, alpha=0.5, label="Fake")
    axs[1].set_title("Channel 2")
    axs[1].set_xlabel("Value")
    axs[1].legend()
    axs[1].grid(True)
    plt.savefig('hist_gp.png', dpi=300)
    plt.show()


def train_WGAN_CNN(G, D, batch_size, n, channel, d_lr, g_lr, is_GP=True, lambda_gp = 10):

    if is_GP:
        d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas = (0,0.9))
        g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas =(0,0.9)) 
    else: 
        d_optimizer = optim.RMSprop(D.parameters(), lr=d_lr)
        g_optimizer = optim.RMSprop(G.parameters(), lr=g_lr)
        
    for _ in range(5):
        D.zero_grad()
        x = torch.randn((batch_size, 2, n))
        x_std = torch.std(x)
        x_norm = x / x_std
        x_norm = x_norm.to(device)

        fake_data = G(x_norm).float().detach()
        ch_input = torch.transpose(x_norm, 1,2).contiguous()
        ch_input = torch.view_as_complex(ch_input)
        real_data = channel(ch_input).detach()
        real_data = torch.view_as_real(real_data)
        real_data = torch.transpose(real_data, 1,2).contiguous()

        kl_loss = KL_approx(fake_data, real_data, x_norm)

        d_real = D(real_data, x_norm)
        d_fake = D(fake_data, x_norm)
        
        if is_GP:
            gradient_penalty = compute_gradient_penalty(D, real_data.data, fake_data.data)
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + lambda_gp * gradient_penalty
        else: 
            d_loss = -(torch.mean(d_real) - torch.mean(d_fake))

        d_loss.backward()
        d_optimizer.step()

        if not is_GP:
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

    # Generator loop
    G.zero_grad()
    x = torch.randn((batch_size,2,n))
    x_std  = torch.std(x)
    x_norm = x/ x_std
    x_norm = x_norm.to(device)

    fake_data = G(x_norm).float()
    g_fake = D(fake_data, x_norm)
    g_loss = -torch.mean(g_fake)

    g_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(G.parameters(), 1.)
    g_optimizer.step()

    return g_loss, d_loss, kl_loss

def train_WGAN(G, D, batch_size, n, channel, d_lr, g_lr, is_GP=False, writer = None, lambda_gp = 10):
    if is_GP:
        d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas = (0,0.9))
        g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas =(0,0.9)) 
    else: 
        d_optimizer = optim.RMSprop(D.parameters(), lr=d_lr)
        g_optimizer = optim.RMSprop(G.parameters(), lr=g_lr)
        
    for _ in range(5):
        D.zero_grad()
        x = torch.randn((batch_size, n))
        x_std = torch.std(x)
        x_norm = x / x_std
        x_norm = x_norm.to(device)

        fake_data = G(x_norm).float().detach()
        real_data = channel(x_norm).detach()
        kl_loss = KL_approx(fake_data, real_data, x_norm)

        d_real = D(real_data, x_norm)
        d_fake = D(fake_data, x_norm)
        
        if is_GP:
            gradient_penalty = compute_gradient_penalty(D, real_data.data, fake_data.data)
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + lambda_gp * gradient_penalty
        else: 
            d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
        d_loss.backward()
        d_optimizer.step()

        if not is_GP:
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

    # Generator loop
    G.zero_grad()
    x = torch.randn((batch_size,n))
    x_std  = torch.std(x)
    x_norm = x/ x_std
    x_norm = x_norm.to(device)

    fake_data = G(x_norm).float()
    g_fake = D(fake_data, x_norm)
    g_loss = -torch.mean(g_fake)

    g_loss.backward()
    g_optimizer.step()

    return kl_loss, g_loss

    

def train_GAN(G, D, batch_size, n, channel, d_lr, g_lr):
    
    d_optimizer = optim.Adam(D.parameters(), lr=d_lr)
    g_optimizer = optim.Adam(G.parameters(), lr=g_lr)
    
    for _ in range(5):
        D.zero_grad()

        x = torch.randn((batch_size,n))
        x_std  = torch.std(x)
        x_norm = x/ x_std
        x_norm = x_norm.detach().to(device)

        fake_data = G(x_norm).float()
        real_data = channel(x_norm).detach()

        d_real = torch.sigmoid(D(real_data, x_norm))
        d_fake = torch.sigmoid(D(fake_data, x_norm))

        d_loss = -torch.mean(torch.log(d_real) + torch.log(1. - d_fake))

        d_loss.backward()
        d_optimizer.step()

        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

    # Generator loop
    G.zero_grad()
    x = torch.randn((batch_size,n))
    x_std  = torch.std(x)
    x_norm = x/ x_std
    x_norm = x_norm.detach().to(device)
    
    fake_data = G(x_norm).float()
    d_fake = torch.sigmoid(D(fake_data, x_norm))
    g_loss = -torch.mean(torch.log(d_fake))

    torch.nn.utils.clip_grad_norm_(G.parameters(), 1.)

    g_loss.backward()
    g_optimizer.step()
            
            
def plot_GAN(G, batch_size, n, channel, epoch):

        x = torch.randn((batch_size, 2, n))
        x_std  = torch.std(x)
        x_norm = x/ x_std
        x_norm = x_norm.detach().to(device)
        test_fakes = torch.flatten(G(x_norm), 1).cpu().detach().numpy()
        x_norm_c = torch.transpose(x_norm, 1,2).contiguous()
        x_norm_c = torch.view_as_complex(x_norm_c)
        test_reals = channel(x_norm_c)
        test_reals = torch.flatten(torch.transpose(torch.view_as_real(test_reals), 1,2).contiguous(), 1)
        test_reals = test_reals.cpu().detach().numpy()

        clear_output(wait=True)
        print("Epoch %s:  " % epoch)
        plt.rcParams["figure.figsize"] = (10,3)
        plt.subplot(1, 2, 1)
        plt.title("Fake Channel")
        plt.scatter(test_fakes[:,0],test_fakes[:,1],s=1)
        plt.subplot(1, 2, 2)
        plt.title("Real Channel")
        plt.scatter(test_reals[:,0],test_reals[:,1],s=1)
        plt.show()


def plot_GAN_m(G, batch_size, M, n, channel, epoch, device):
        
        if n==2:
            msgs = torch.Tensor([m] * batch_size).to(device) #torch.randint(M, size=(batch_size,1)).to(device)
            print(msgs.shape)
            msgs = msgs.reshape((batch_size,-1))
            pseudocode = pseudo_qcode(msgs)
            test_fakes = G(pseudocode.float()).cpu().detach().numpy()
            test_reals =  channel(pseudocode.float()).cpu().detach().numpy()
        else:
            
            x = torch.randn(n)
            x = x.repeat(batch_size).reshape(batch_size,n)
            x = x.to(device)
            """
            msgs = torch.Tensor([m] * batch_size).to(device) #torch.randint(M, size=(batch_size,1)).to(device)
            print(msgs.shape)
            msgs = msgs.reshape((batch_size,-1))
            pseudocode = pseudo_qcode(msgs)
            
            x = torch.randn((batch_size,n))
            x_std  = torch.std(x)
            x_norm = x/ x_std
            x_norm = x_norm.detach().to(device)
            """
            test_fakes = G(x).cpu().detach().numpy()
            test_reals =  channel(x).cpu().detach().numpy()
            
        clear_output(wait=True)
        print("Epoch %s:  " % epoch)
        plt.rcParams["figure.figsize"] = (10,3)
        plt.subplot(1, 2, 1)
        plt.title("Fake Channel")
        plt.scatter(test_fakes[:,0],test_fakes[:,1],s=0.5)
        plt.subplot(1, 2, 2)
        plt.title("Real Channel")
        plt.scatter(test_reals[:,0],test_reals[:,1],s=0.5)
        plt.show()


def KL_approx(fake_data, real_data, inputs):
    
    def kl_div_approx(p, q):
        w = 1e-5
        p = p + w
        q = q + w
        return torch.sum(p * torch.log(p / q))
    
    fdata = torch.reshape(fake_data-inputs, (-1,))
    rdata = torch.reshape(real_data-inputs, (-1,))

    rdata_bin = torch.histc(rdata.cpu(), bins=100, min=-3, max=3)
    fdata_bin = torch.histc(fdata.cpu(), bins=100, min=-3, max=3)

    q = rdata_bin / torch.sum(rdata_bin)
    p = fdata_bin / torch.sum(fdata_bin)

    return kl_div_approx(p,q)



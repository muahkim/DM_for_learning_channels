import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Channel for both
    
class Channel(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        channel_noise = noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = codeword + channel_noise
        dec_signal = self.dec(rec_signal)

        return dec_signal
    
class Channel_AWGN(torch.nn.Module):
    def __init__(self, noise_std):
        super(Channel_AWGN, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.noise_std = noise_std

    def forward(self, inputs):
        #End-to-End Framework
        
        i,j = inputs.size()
        channel_noise = self.noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = inputs + channel_noise

        return rec_signal
    
    
class Channel_GAN(torch.nn.Module):
    def __init__(self, enc, dec, channel):
        super(Channel_GAN, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
        self.channel = channel

    def forward(self, inputs):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        rec_signal = self.channel(codeword)
        dec_signal = self.dec(rec_signal)

        return dec_signal
    
class Channel_ray_only(torch.nn.Module):
    def __init__(self, noise_std):
        super(Channel_ray_only, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.noise_std = noise_std

    def forward(self, inputs):
        #End-to-End Framework
        
        i,j = inputs.size()
        
        h_ray = (1/np.sqrt(2))*torch.sqrt(torch.randn((i, j)).to(self.device)**2 + torch.randn((i, j)).to(self.device)**2)
        channel_noise = self.noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = h_ray*inputs + channel_noise

        return rec_signal
    
    
class Channel_ray(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_ray, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        h_ray = (1/np.sqrt(2))*torch.sqrt(torch.randn((i, j)).to(self.device)**2 + torch.randn((i, j)).to(self.device)**2)
        channel_noise = noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = h_ray*codeword + channel_noise
        dec_signal = self.dec(rec_signal)

        return dec_signal
    
class Channel_burst(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_burst, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        channel_noise = noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        burst = np.sqrt(2)*channel_noise
        burst_prob = torch.bernoulli(torch.tensor(.1)).to(self.device)
        
        rec_signal = codeword + channel_noise + burst*burst_prob
        dec_signal = self.dec(rec_signal)

        return dec_signal


def ch_SSPA(x, sigma_n, device, p = 3.0, A_0 = 1.5, v = 5.0): # A_0 limiting output amplitude, v is small signal gain.
    assert x.size(1) % 2 == 0

    # x= ([1 2 3 4] ,[5 6 7 8])
    dim = int(x.size(1) //2)
    x_2d = x.reshape(-1,2) #x_2d =([1 2], [3 4] , [5 6], [7 8])
    A = torch.sum(x_2d ** 2, dim=1) ** 0.5 # Amplitude
    A_mean = torch.mean(A)
    #print(f'Mean amplitude of the signals: {A_mean: .2f}')
    A_ratio = v / (1+ (v*A/A_0)**(2*p) )**(1/2/p) # g_A / A
    x_amp_2d = torch.mul(A_ratio.reshape(-1,1), x_2d)
    x_amp = x_amp_2d.reshape(-1, 2*dim)
    A_amp_mean = torch.mean(torch.mul(A_ratio,A))
    #print(f'Mean amplitude of the amplified signals: {A_amp_mean : .2f}')
    y = x_amp + 2 **0.5* sigma_n * torch.randn_like(x)
    
    return y


class Channel_SSPA_only(torch.nn.Module):
    def __init__(self, noise_std):
        super(Channel_SSPA_only, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.noise_std = noise_std

    def forward(self, inputs):
        #End-to-End Framework
        rec_signal = ch_SSPA(inputs, self.noise_std, self.device)
        return rec_signal
    
class Channel_SSPA(torch.nn.Module):
    def __init__(self, enc, dec, use_cuda=True):
        super(Channel_SSPA, self).__init__()
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = "cpu"
        else: self.device = "cpu"
        #use_cuda = torch.cuda.is_available()
        #self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
        
    def forward(self, inputs, noise_std):
        
        codeword = self.enc(inputs)

        rec_signal = ch_SSPA(codeword, noise_std, self.device)
        dec_signal = self.dec(rec_signal)

        return dec_signal


class Channel_GAN_SSPA(torch.nn.Module):
    def __init__(self, enc, dec, channel, use_cuda=True):
        super(Channel_GAN_SSPA, self).__init__()
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        else:
            self.device = "cpu"
        #use_cuda = torch.cuda.is_available()
        #self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
        self.channel = channel
    def forward(self, inputs, noise_std):
        codeword = self.enc(inputs)
        rec_signal = self.channel(codeword)
        dec_signal = self.dec(rec_signal)

        return dec_signal


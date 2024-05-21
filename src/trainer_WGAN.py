import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils


import model_WGAN as model




class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 500
    M=16
    learning_rate = 1e-3
    rate = 4/7
    num_workers = 0 # for DataLoader
    TRAINING_EbN0 = 6
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, config, one_hot):
        self.model = model
        self.train_dataset = train_dataset
        #self.test_dataset = test_dataset
        self.config = config
        self.one_hot = one_hot

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)



    def test(self, snr_range, one_hot, erasure_bound):
        model, config = self.model, self.config
        ser = []
                
        def run_epoch():
            data = self.train_dataset
            batch_size = config.batch_size
            M = config.M
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)            
            batch_BER = [] 
            block_ER = []
            erasures = []
            batch_SER = []
            
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5, ascii=True)
            
            for it, x in pbar:
                if self.one_hot:
                    labels = x.reshape((batch_size,1)).to(self.device)
                    x=F.one_hot(labels.long(), num_classes=M)
                    x = x.float().reshape(batch_size,M)
                else:
                    x = x.float().to(self.device)
                noise_std_ = utils.EbNo_to_noise(snr_range[epoch], config.rate)
                #noise_std_ = utils.SNR_to_noise(snr_range[epoch])
                output = model(x, noise_std_)
                
                if self.one_hot:
                    SER, _, _ = utils.SER(labels.long(), output, erasure_bound)
                    batch_SER.append(SER.item())
                    #block_ER.append(bler)
                    #erasures.append(Erasures)
                    if it%100==0:
                        pbar.set_description(f"SNR {snr_range[epoch]}  iter {it}: SER {np.mean(batch_SER):.3e}")
                else:
                    batch_BER.append(utils.B_Ber_m(x.float(), output).item())
                    block_ER.append(utils.Block_ER(x.float(), output))
                    pbar.set_description(f"SNR {snr_range[epoch]} iter {it}: avg. BER {np.mean(batch_BER):.8f} BLER {np.mean(block_ER):.7e}")
       
            return np.mean(batch_SER)
        
        num_samples = self.train_dataset.shape[0]
        for epoch in range(len(snr_range)):

            temp1 = 0
            it_in_while = 0

            while temp1 * num_samples *(it_in_while) < 5000: # To guarantee we have enough samples for the Monte-Carlo method.
                if it_in_while > 1 :
                    print("The number of samples is not enough. One more epoch is running. Total # of samples used: " , num_samples * it_in_while)
                temp2 = run_epoch()
                temp1 = (it_in_while * temp1 + temp2)/(it_in_while+1) # taking average of the error probability.
                it_in_while += 1
            ser.append(temp1)
        
        return ser
    
    def train(self,  weights, GAN=False, writer=None, global_epoch=0):
        config =  self.config
        model = self.model 
        optimizer = torch.optim.NAdam(weights, lr=config.learning_rate)

        loss_CE = nn.CrossEntropyLoss()
        
        def run_epoch():
            lr = config.learning_rate
            batch_size = config.batch_size
            M = config.M
            data = self.train_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)            
            losses, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5, ascii=True)
            
            for it, x in pbar:
                labels = x.reshape((batch_size,1)).to(self.device)
                x=F.one_hot(labels.long(), num_classes=M)                    
                x = x.float().reshape(batch_size,M)

                optimizer.zero_grad()
                if GAN==True:
                    output = model(x)
                else:
                    output = model(x, config.noise_std)
                    
                if self.one_hot:
                    loss = loss_CE(output, labels.long().squeeze(1))
                else:
                    loss =  F.binary_cross_entropy(output, x)
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(weights, 1.)
                losses.append(loss.item())
                optimizer.step()
                
                if self.one_hot:
                    SER, _, _ = utils.SER(labels.long(), output, 0.9)
                    batch_SER.append(SER.item())
                    if it%100==0:
                        pbar.set_description(f"epoch {global_epoch+epoch+1}: loss {np.mean(losses):.2e} SER {np.mean(batch_SER):.2e}")
                else:
                    batch_BER.append(utils.B_Ber_m(x, output).item())
                    pbar.set_description(f"epoch {global_epoch+epoch+1}: loss {np.mean(losses):.2e} BER {np.mean(batch_BER):.2e}")          

                if writer is not None:
                    writer.add_scalar('loss/AE train', np.mean(losses), global_epoch+epoch+1)
                    writer.add_scalar('SER/AE train', np.mean(batch_SER), global_epoch+epoch+1)
                    
            
        for epoch in range(config.max_epochs):
            run_epoch()
        return 




class Trainer_:
    def __init__(self, encoder, decoder, G, train_dataset, config, one_hot, writer=None):
        #self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.G = G
        self.train_dataset = train_dataset
        #self.test_dataset = test_dataset
        self.config = config
        self.one_hot = one_hot
        self.writer = writer
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
#            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def test(self, snr_range, erasure_bound):
        config = self.config
        model_val = model.Channel_SSPA(self.encoder, self.decoder)
        #noise_std_ = utils.EbNo_to_noise(config.TRAINING_EbN0, config.rate)
        
        def run_epoch():
            data = self.train_dataset
            batch_size = config.batch_size
            M = config.M
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers, drop_last=True)
            batch_BER = []
            block_ER = []
            erasures = []
            batch_SER = []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5, ascii=True)

            for it, x in pbar:
                if self.one_hot:
                    labels = x.reshape((batch_size, 1)).to(self.device)
                    x = F.one_hot(labels.long(), num_classes=M)
                    x = x.float().reshape(batch_size, M)
                else:
                    x = x.float().to(self.device)

                noise_std_ = utils.EbNo_to_noise(snr_range[epoch],config.rate)
                output = model_val(x, noise_std_)

                if self.one_hot:
                    SER, _, _ = utils.SER(labels.long(), output, erasure_bound)
                    batch_SER.append(SER.item())
                    # block_ER.append(bler)
                    # erasures.append(Erasures)
                    if it % 100 == 0:
                        pbar.set_description(f"SNR {snr_range[epoch]}  iter {it}: SER {np.mean(batch_SER):.3e}")
                else:
                    batch_BER.append(utils.B_Ber_m(x.float(), output).item())
                    block_ER.append(utils.Block_ER(x.float(), output))
                    pbar.set_description(
                        f"SNR {snr_range[epoch]} iter {it}: avg. BER {np.mean(batch_BER):.8f} BLER {np.mean(block_ER):.7e}")

            return np.mean(batch_SER)


        num_samples = self.train_dataset.shape[0]
        ser=[]
        for epoch in range(len(snr_range)):

            temp1 = 0
            it_in_while = 0

            while temp1 * num_samples * (it_in_while) < 1000:  # To guarantee we have enough samples for the Monte-Carlo method.
                if it_in_while > 1:
                    print("The number of samples is not enough. One more epoch is running. Total # of samples used: ",
                          num_samples * it_in_while)
                temp2 = run_epoch()
                temp1 = (it_in_while * temp1 + temp2) / (it_in_while + 1)  # taking average of the error probability.
                it_in_while += 1
            ser.append(temp1)
        return ser

    def train(self,  weights, GAN=False, global_epoch=0):
        config = self.config
        writer = self.writer
        model_tr=model.Channel_GAN(self.encoder, self.decoder, self.G).to(self.device)
        if torch.cuda.is_available():
            model_tr = torch.nn.DataParallel(model_tr).to(self.device)

        #model = self.model
        optimizer = torch.optim.NAdam(weights, lr=config.learning_rate)

        loss_CE = nn.CrossEntropyLoss()
        
        def run_epoch_tr():
            lr = config.learning_rate
            batch_size = config.batch_size
            M = config.M
            data = self.train_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)            
            losses, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5, ascii=True)
            
            for it, x in pbar:
                labels = x.reshape((batch_size,1)).to(self.device)
                x=F.one_hot(labels.long(), num_classes=M)                    
                x = x.float().reshape(batch_size,M)

                optimizer.zero_grad()
                if GAN==True:
                    output = model_tr(x)
                else:
                    output = model_tr(x, config.noise_std)
                    
                if self.one_hot:
                    loss = loss_CE(output, labels.long().squeeze(1))
                else:
                    loss =  F.binary_cross_entropy(output, x)
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(weights, 1.)
                losses.append(loss.item())
                optimizer.step()
                
                if self.one_hot:
                    SER, _, _ = utils.SER(labels.long(), output, 0.9)
                    batch_SER.append(SER.item())
                    if it%100==0:
                        pbar.set_description(f"epoch {global_epoch+epoch+1}: loss {np.mean(losses):.2e} SER {np.mean(batch_SER):.2e}")
                else:
                    batch_BER.append(utils.B_Ber_m(x, output).item())
                    pbar.set_description(f"epoch {global_epoch+epoch+1}: loss {np.mean(losses):.2e} BER {np.mean(batch_BER):.2e}")          

                if writer is not None:
                    writer.add_scalar('loss/AE train', np.mean(losses), global_epoch+epoch+1)
                    writer.add_scalar('SER/AE train', np.mean(batch_SER), global_epoch+epoch+1)
                    
            
        for epoch in range(config.max_epochs):
            run_epoch_tr()
            val_SER=self.test( [self.config.TRAINING_EbN0], erasure_bound=0.7)
            if writer is not None:
                writer.add_scalar('SER/validation', val_SER[0], global_epoch+epoch+1)
            print(f"epoch {global_epoch+epoch+1}: validation SER {val_SER[0]:.2e}")
        return 
import numpy as np
import torch
import os
from .base_model import BaseModel
from . import networks
import math


class DynaAWGNModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out
        self.loss_names = ['G_L2', 'G_reward']
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'fake', 'real_B']
        # specify the models you want to save
        self.model_names = ['SE', 'CE', 'G', 'P']

        # define networks
        self.netSE = networks.define_SE(input_nc=opt.input_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                        n_downsample=opt.n_downsample, norm=opt.norm, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netCE = networks.define_CE(ngf=opt.ngf, max_ngf=opt.max_ngf, n_downsample=opt.n_downsample,
                                        C_channel=opt.C_channel, norm=opt.norm, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.netG = networks.define_dynaG(output_nc=opt.output_nc, ngf=opt.ngf, max_ngf=opt.max_ngf,
                                          n_downsample=opt.n_downsample, C_channel=opt.C_channel,
                                          n_blocks=opt.n_blocks, norm=opt.norm, init_type=opt.init_type,
                                          init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.netP = networks.define_dynaP(ngf=opt.ngf, max_ngf=opt.max_ngf,
                                          n_downsample=opt.n_downsample, init_type=opt.init_type,
                                          init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                          N_output=opt.G_s + 1)

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            params = list(self.netSE.parameters()) + list(self.netCE.parameters()) + list(
                self.netG.parameters()) + list(self.netP.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr_joint, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.opt = opt
        self.temp = opt.temp_init if opt.isTrain else 5

    def name(self):
        return 'DynaAWGN_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def forward(self):
        # Generate SNR
        self.snr = torch.rand(self.real_A.shape[0], 1).to(self.device) * (
            self.opt.SNR_MAX - self.opt.SNR_MIN) + self.opt.SNR_MIN if self.opt.isTrain else \
            torch.ones(self.real_A.shape[0], 1).to(self.device) * self.opt.SNR

        # Generate latent vector
        z = self.netSE(self.real_A)
        latent = self.netCE(z, self.snr)

        # Generate decision mask
        self.hard_mask, self.soft_mask, prob = self.netP(z, self.snr, self.temp)
        self.count = self.hard_mask.sum(-1)

        # Normalize each channel
        latent_sum = torch.sqrt((latent ** 2).mean((-2, -1), keepdim=True))
        latent = latent / latent_sum

        # Generate the full mask
        N, C, W, H = latent.shape
        pad = torch.ones((N, self.opt.G_n), device=self.device)
        self.hard_mask = torch.cat((pad, self.hard_mask), -1)
        self.soft_mask = torch.cat((pad, self.soft_mask), -1)

        latent_res = latent.view(N, self.opt.G_s + self.opt.G_n, -1)

        # Selection with either soft mask or hard mask
        mask = self.soft_mask if self.opt.select == 'soft' and self.opt.isTrain else self.hard_mask
        latent_res = latent_res * mask.unsqueeze(-1)

        # Pass through the AWGN channel
        with torch.no_grad():
            sigma = 10 ** (-self.snr / 20)
            noise = sigma.view(self.real_A.shape[0], 1, 1) * torch.randn_like(latent_res)
            noise = noise * mask.unsqueeze(-1)
        latent_res = latent_res + noise

        # Here is the change: call netG with the updated latent_res and SNR
        self.fake = self.netG(latent_res.view(latent.shape), self.snr)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B)
        self.loss_G_reward = torch.mean(self.count)
        self.loss_G = self.opt.lambda_L2 * self.loss_G_L2 + self.opt.lambda_reward * self.loss_G_reward
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

    def update_temp(self):
        self.temp *= math.exp(-self.opt.eta)
        self.temp = max(self.temp, 0.005)

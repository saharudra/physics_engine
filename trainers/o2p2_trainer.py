import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.utils import save_image

from modules.vgg import Vgg16
from models.o2p2 import O2P2Model

import numpy as np  
import argparse 
from tqdm import trange
from misc.utils import *


class O2P2Trainer(nn.Module):
    def __init__(self, params, model, train_loader, val_loader, logger, results_path, logs_path):
        super(O2P2Trainer, self).__init__()
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.results_path = results_path
        self.logs_path = logs_path
        self.vgg = Vgg16(requires_grad=False)
        if self.params['use_cuda']:
            self.vgg = self.vgg.cuda()
        self.train_iteration = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

    def compute_perceptual_loss(self, img1, img2):
        features_img1 = self.vgg(img1)
        features_img2 = self.vgg(img2)
        loss_criterion = nn.MSELoss()
        return loss_criterion(features_img1.relu2_2, features_img2.relu2_2)

    def compute_pixel_loss(self, img1, img2):
        if self.params['pixel_loss_type'] == 'l1':
            return torch.mean(torch.abs(img1 - img2))
        elif self.params['pixel_loss_type'] == 'l2':
            loss_criterion = nn.MSELoss()
            return loss_criterion(img1, img2)

    def train(self):
        for epoch in range(self.params['max_epochs']):
            with trange(len(self.train_loader)) as t:
                self.model.train()
                for idx, sample in enumerate(self.train_loader):
                    # Get the required inputs to the model
                    ini_img, fin_img, ini_masks, num_objs = sample['ini_img'], sample['fin_img'], sample['ini_masks'], sample['num_objs']

                    # Place them on cuda resources
                    if self.params['use_cuda']:
                        ini_img = ini_img.cuda()
                        fin_img = fin_img.cuda()
                        ini_masks = ini_masks.cuda()
                        num_objs = num_objs.cuda()
                    
                    # Create loss dict to add losses to logger
                    loss_dict = {}

                    # Zero the gradient and perform forward pass with the model
                    self.optimizer.zero_grad()
                    recon_ini_img, recon_fin_img = self.model(ini_img, ini_masks, num_objs)

                    # Calculate losses and perform the backward pass
                    pixel_loss_ini_img = self.compute_pixel_loss(ini_img, recon_ini_img)
                    pixel_loss_fin_img = self.compute_pixel_loss(fin_img, recon_fin_img)
                    percept_loss_ini_img = self.compute_perceptual_loss(ini_img, recon_ini_img)
                    percept_loss_fin_img = self.compute_perceptual_loss(fin_img, recon_fin_img)
                    overall_loss = pixel_loss_ini_img + pixel_loss_fin_img + \
                                   percept_loss_ini_img + percept_loss_fin_img

                    overall_loss.backward()
                    self.optimizer.step()

                    loss_dict = info_dict('pixel_loss_ini_img', pixel_loss_ini_img.item(), loss_dict)
                    loss_dict = info_dict('pixel_loss_fin_img', pixel_loss_fin_img.item(), loss_dict)
                    loss_dict = info_dict('percept_loss_ini_img', percept_loss_ini_img.item(), loss_dict)
                    loss_dict = info_dict('percept_loss_fin_img', percept_loss_fin_img.item(), loss_dict)
                    loss_dict = info_dict('overall_loss', overall_loss.item(), loss_dict)

                    # Add losses to logger
                    for tag, value in loss_dict.items():
                        self.logger.scalar_summary(tag, value, self.train_iteration)

                    # Save original and reconstructed images
                    save_image(recon_ini_img, self.results_path + 'epoch_' + str(epoch) + '_reconstructed_ini_img.jpg')
                    save_image(ini_img, self.results_path + 'epoch_' + str(epoch) + '_ini_img.jpg')
                    save_image(recon_fin_img, self.results_path + 'epoch_' + str(epoch) + '_reconstructed_fin_img.jpg')
                    save_image(fin_img, self.results_path + 'epoch_' + str(epoch) + '_fin_img.jpg')
                    
                    self.train_iteration += 1
                    loss_dict = info_dict('Epoch', epoch, loss_dict)
                    t.set_postfix(loss_dict)
                    t.update()
            
            # Save model and run validation
            if epoch % self.params['loggin_interval'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': overall_loss
                }, self.logs_path + 'epoch_' + str(epoch) + '_model.pth')
                self.eval(self.train_iteration, epoch)

        return loss_dict['overall_loss'], self.train_iteration


    def eval(self, iteration, epoch):
        self.model.eval()
        for idx, sample in enumerate(self.val_loader):
            # Get the required inputs to the model
            ini_img, fin_img, ini_masks, num_objs = sample['ini_img'], sample['fin_img'], sample['ini_masks'], sample['num_objs']

            # Place them on cuda resources
            if self.params['use_cuda']:
                ini_img = ini_img.cuda()
                fin_img = fin_img.cuda()
                ini_masks = ini_masks.cuda()
                num_objs = num_objs.cuda()
            
            # Create loss dict to add losses to logger
            loss_dict = {}
            recon_ini_img, recon_fin_img = self.model(ini_img, ini_masks, num_objs)

            # Calculate losses and perform the backward pass
            pixel_loss_ini_img = self.compute_pixel_loss(ini_img, recon_ini_img)
            pixel_loss_fin_img = self.compute_pixel_loss(fin_img, recon_fin_img)
            percept_loss_ini_img = self.compute_perceptual_loss(ini_img, recon_ini_img)
            percept_loss_fin_img = self.compute_perceptual_loss(fin_img, recon_fin_img)
            overall_loss = pixel_loss_ini_img + pixel_loss_fin_img + \
                            percept_loss_ini_img + percept_loss_fin_img

            loss_dict = info_dict('val_pixel_loss_ini_img', pixel_loss_ini_img.item(), loss_dict)
            loss_dict = info_dict('val_pixel_loss_fin_img', pixel_loss_fin_img.item(), loss_dict)
            loss_dict = info_dict('val_percept_loss_ini_img', percept_loss_ini_img.item(), loss_dict)
            loss_dict = info_dict('val_percept_loss_fin_img', percept_loss_fin_img.item(), loss_dict)
            loss_dict = info_dict('val_overall_loss', overall_loss.item(), loss_dict)

            # Add losses to logger
            for tag, value in loss_dict.items():
                self.logger.scalar_summary(tag, value, iteration)
            
        save_image(recon_ini_img, self.results_path + 'epoch_' + str(epoch) + '_val_reconstructed_ini_img.jpg')
        save_image(recon_fin_img, self.results_path + 'epoch_' + str(epoch) + '_val_reconstructed_fin_img.jpg')
        save_image(ini_img, self.results_path + 'epoch_' + str(epoch) + '_val_ini_img.jpg')
        save_image(fin_img, self.results_path + 'epoch_' + str(epoch) + '_val_fin_img.jpg')

        
                    

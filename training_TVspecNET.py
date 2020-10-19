# -------------------------------------------------------------------------------------------------------------------------------------------
# @article{grossmann2020,
#    title={Deeply Learned Spectral Total Variation Decomposition},
#    author={Tamara G. Grossmann and Yury Korolev and Guy Gilboa and Carola-Bibiane Schönlieb},
#    year={2020},
#    eprint={2006.10004},
#    archivePrefix={arXiv},
#    primaryClass={cs.CV}
#  }
#
# Author: Tamara G. Grossmann (2020), tg410@cam.ac.uk
# This code is partially taken from the DnCNN implementation from Kai Zhang (2018) https://github.com/cszn/DnCNN and amended for our purpose. 
# The U-Net basis architecture for U-TVspecNET is taken from Naoto Usuyama (2018) https://github.com/usuyama/pytorch-unet/, and 
# the FFDNet basis architecture for F-TVspecNET is taken from Kai Zhang (2019) https://github.com/cszn/KAIR
#
# The following code is to train the TVspecNET model on .mat data type images (see data generator). The default loss functional is set to the 
# normalised MSE between the ground truth (GT) and the network output. However, we include code for adding the normalised MSE of the image 
# gradients of the GT and network output, and the MSE between the input image and the sum over all output images to the loss functional. 
# Please uncomment the corresponding lines (see ll.414-419, ll.440-445, ll.474-479, ll.499-504, ll.537-543)
#
# We additionally enable training of F-TVspecNET (choose --model 'F-TVspecNET') and U-TVspecNET (choose --model 'U-TVspecNET').
#
# PyTorch Version 1.1.0 used for this implementation. 
# TensorboardX is used for visualisation during training of both the plot of the loss function and the network outputs during training.
# --------------------------------------------------------------------------------------------------------------------------------------------


import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models # for loading weights of a pretrained model for U-TVspecNET initialisation
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter # for visualisation of loss and images during training
import data_generator_decomp_dyadic as dg # data generator for training data
import data_generator_decomp_test_dyadic as dgt # data generator for testing data
from data_generator_decomp_dyadic import DecompositionDataset 


# ============================================================================================================================
# 1. Get parser arguments
# ============================================================================================================================
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model_id', default='TVspecNET', type=str, help='choose an id for model')
parser.add_argument('--model', default='TVspecNET', type=str, help='choose a type of model: TVspecNET, FTVspecNET or UTVspecNET')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--train_data', default='data/train', type=str, help='path of train data')
parser.add_argument('--test_data', default='data/train', type=str, help='path of test data')
parser.add_argument('--epoch', default=5000, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')#1e-3
parser.add_argument('--cuda_card', default=0, type=int, help='select the cuda card this is supposed to run on')
parser.add_argument('--summary_dir', default='./summary/TVspecNET_training', type=str, help='set the location to store the summary info')

args = parser.parse_args()
batch_size = args.batch_size
n_epoch = args.epoch
model_name = args.model

# Select the correct cuda card
cuda = torch.cuda.is_available()
cuda1 = torch.device('cuda',args.cuda_card)

save_dir = os.path.join('models', args.model_id)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
      
# ============================================================================================================================
# 2. Define the model architectures a) TVspecNET, b) F-TVspecNET, c) U-TVspecNET
# ============================================================================================================================  
    
#-------------------------------------------------------------------------------------------------------------------
# a) TVspecNET (with DnCNN as the basis architecture)
#-------------------------------------------------------------------------------------------------------------------    
class TVspecNET(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, band_channels = 5, use_bnorm=True, kernel_size=3): 
        super(TVspecNET, self).__init__()
        kernel_size = 3 
        padding = 1 
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False)) 
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=band_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)    
                
                
#-------------------------------------------------------------------------------------------------------------------
# b) F-TVspecNET with FFDnet as the basis architecture
#-------------------------------------------------------------------------------------------------------------------
def PixelUnShuffle(input, factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(batch_size, channels, out_height, factor,out_width, factor)

    channels *= factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class FTVspecNET(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, band_channels = 5, use_bnorm=True, kernel_size=3):
        super(FTVspecNET, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        sf = 2
        
        self.m_up = nn.PixelShuffle(upscale_factor=sf)

        layers.append(nn.Conv2d(in_channels=image_channels*sf*sf, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=band_channels*sf*sf, kernel_size=kernel_size, padding=padding, bias=False))
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = PixelUnShuffle(x,2)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]
        return x      

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
        
#-------------------------------------------------------------------------------------------------------------------
# c) U-TVspecNET (with U-Net as the basis architecture)
#-------------------------------------------------------------------------------------------------------------------
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UTVspecNET(nn.Module):
    def __init__(self, image_channels=1, band_channels=5, kernel_size=3): 
        super(UTVspecNET, self).__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())       
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, band_channels, 1)
        
    def forward(self, x):
        x_original = self.conv_original_size0(x.repeat(1,3,1,1))
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(x.repeat(1,3,1,1))            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
 
        return out


# ============================================================================================================================
# 3. Define loss functionals: a) MSE, b) Huber norm of image gradients, c) MSE of sum over bands
# ============================================================================================================================  
#-------------------------------------------------------------------------------------------------------------------
# a) MSE
#-------------------------------------------------------------------------------------------------------------------
class MSE_normalised(torch.nn.Module):
    def __init__(self):
        super(MSE_normalised, self).__init__()

    def forward(self, input, target):     
        return torch.mean(torch.mean(torch.mean((input-target)**2,-2),-1)/(torch.mean(torch.mean(target**2,-2),-1)+1e-4))
    
#-------------------------------------------------------------------------------------------------------------------
# b) Huber norm of image gradients 
#-------------------------------------------------------------------------------------------------------------------
def image_gradient(input):
    kernelAx = torch.FloatTensor([[1,-1]])
    kernelAx = kernelAx[None,None].to(cuda1)
    kernelAy = torch.FloatTensor([[1],[-1]])
    kernelAy = kernelAy[None,None].to(cuda1)
    mx = nn.ReflectionPad2d((1,0,0,0))
    my = nn.ReflectionPad2d((0,0,1,0))
    img_grad_x = input.new_empty(input.shape)
    img_grad_y = input.new_empty(input.shape)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            img_grad_x[i,j,:,:] = nn.functional.conv2d(mx(input)[i,j,:,:][None,None],kernelAx)
            img_grad_y[i,j,:,:] = nn.functional.conv2d(my(input)[i,j,:,:][None,None],kernelAy)
    return img_grad_x, img_grad_y

class Huber_normalised(torch.nn.Module):
    def __init__(self):
        super(Huber_normalised, self).__init__()
        self.huber_loss = torch.nn.SmoothL1Loss()

    def forward(self, input, target):
        img_grad_x, img_grad_y = image_gradient(input)
        gt_grad_x, gt_grad_y = image_gradient(target)
        img_grad = torch.cat((img_grad_x,img_grad_y),2)
        gt_grad = torch.cat((gt_grad_x,gt_grad_y),2)

        zero_tensor = torch.zeros([gt_grad.shape[2],gt_grad.shape[3]]).to(cuda1)
        gt_temp = torch.zeros([gt_grad.shape[0],gt_grad.shape[1]])
        Huber_temp = torch.zeros([gt_grad.shape[0],gt_grad.shape[1]])

        for i in range(0,input.shape[0]):
            for j in range(0,input.shape[1]):
                Huber_temp[i,j] = self.huber_loss(img_grad[i,j,:,:], gt_grad[i,j,:,:])
                gt_temp[i,j] = self.huber_loss(gt_grad[i,j,:,:], zero_tensor)
        out = torch.mean(Huber_temp.to(cuda1)/(gt_temp.to(cuda1)+1e-4))
        return out

#-------------------------------------------------------------------------------------------------------------------
# c) MSE of sum over bands
#-------------------------------------------------------------------------------------------------------------------
class MSE_bands_normalised(torch.nn.Module):
    def __init__(self):
        super(MSE_bands_normalised, self).__init__()

    def forward(self, input, target):
        addition = torch.sum(input,1,keepdim=True) 
        out = torch.mean(torch.mean(torch.mean((addition-target)**2,-2),-1)/(torch.mean(torch.mean(target**2,-2),-1)+1e-4))
        return out

# ============================================================================================================================
# 4. Find last Checkpoint and set initial epoch accordingly
# ============================================================================================================================  
def findLastCheckpoint(save_dir):
    file_list = os.path.join(save_dir, 'model_save.pth')
    if os.path.isfile(file_list):
        checkpoint = torch.load(file_list)
        initial_epoch = checkpoint['epoch']
    else:
        initial_epoch = 0
        checkpoint = 0
    return initial_epoch, checkpoint


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


# ============================================================================================================================
# 5. Main training
# ============================================================================================================================  
if __name__ == '__main__':
    torch.manual_seed(28) # to obtain reproducability 
    writer = SummaryWriter(os.path.join('summary/',args.summary_dir)) # to get a tensorboardX visual output during training
    
    # model selection
    print('===> Building model')
    if model_name == 'TVspecNET':
        model = TVspecNET()
    elif model_name == 'FTVspecNET':
        model = FTVspecNET()
    elif model_name == 'UTVspecNET':
        model = UTVspecNET()
    else:
        print('Error, model name not existent.')
    
    # Loss functionals
    criterion1 = MSE_normalised() #default
    criterion2 = Huber_normalised() #MSE of image gradients
    criterion3 = MSE_bands_normalised() #MSE of sum over bands and input image

    if cuda:
        model = model.to(cuda1)
    
    # Choose optimiser, scheduler, etc.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.2)  # learning rates change at epochs 30, 60, and 90
    losses_train_test = []
    losses_separated = []
    losses_bands = []
    test_losses_separated = []
    
    initial_epoch, checkpoint = findLastCheckpoint(save_dir=save_dir)  # load the last model 
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.train() #training mode

    #-----------------------------------------------------------------------------------------------------------------
    # Load Data:
    #-----------------------------------------------------------------------------------------------------------------
    xs, xs_info = dg.datagenerator(data_dir=args.train_data)
    xs = xs.copy()
    xs_info = xs_info.copy()
    xs_info = np.ascontiguousarray(xs_info)
    xs = torch.from_numpy(xs) # NXHXW
    DDataset = DecompositionDataset(xs, xs_info)
    DLoader = DataLoader(dataset=DDataset, num_workers=8, drop_last=True, batch_size=batch_size, shuffle=True)

    test_xs, test_info = dgt.datagenerator(data_dir=args.test_data) # testing data to run in parallel and check performance
    test_xs = test_xs.copy()
    test_info = test_info.copy()
    test_info = np.ascontiguousarray(test_info)
    test_xs = torch.from_numpy(test_xs)
    testDDataset = DecompositionDataset(test_xs, test_info)
    testDLoader = DataLoader(dataset=testDDataset, num_workers=8, drop_last=True, batch_size=batch_size, shuffle=True)
    
    #-----------------------------------------------------------------------------------------------------------------
    # Training:
    #-----------------------------------------------------------------------------------------------------------------
    for epoch in range(initial_epoch, n_epoch):

        scheduler.step(epoch)  # step to the learning rate in this epoch
        
        start_time = time.time()
        model.train()
        epoch_loss = 0
        epoch_band1 = 0
        epoch_band2 = 0
        epoch_band3 = 0
        epoch_band4 = 0
        epoch_band5 = 0
        epoch_band6 = 0
  
        # In case of multiple loss functions the following scalars are to track the individual losses (in train and test mode)
        #epoch_grad = 0
        #epoch_mse = 0
        #epoch_add = 0
        #tepoch_grad = 0
        #tepoch_mse = 0
        #tepoch_add = 0
        test_epoch_loss = 0

        for n_count, batch_yx in enumerate(DLoader):
                optimizer.zero_grad()
                if cuda:
                    batch_x, batch_y = batch_yx[0].to(cuda1), batch_yx[1].to(cuda1)    

                res_band = batch_x - torch.sum(batch_y,1,keepdim=True) #residual band

                loss = criterion1(model(batch_x), batch_y) #+ criterion2(model(batch_x), batch_y) + criterion3(model(batch_x),batch_x - res_band) 

                # Tracking the loss for each individual band
                band1 = criterion1(model(batch_x)[:,0,:,:], batch_y[:,0,:,:])
                band2 = criterion1(model(batch_x)[:,1,:,:], batch_y[:,1,:,:]) 
                band3 = criterion1(model(batch_x)[:,2,:,:], batch_y[:,2,:,:])
                band4 = criterion1(model(batch_x)[:,3,:,:], batch_y[:,3,:,:])
                band5 = criterion1(model(batch_x)[:,4,:,:], batch_y[:,4,:,:])
                new_band6 = batch_x-torch.sum(model(batch_x),1,keepdim=True)
                band6 = criterion1(new_band6, res_band)

#                 msel = criterion1(model(batch_x), batch_y)
#                 gra = criterion2(model(batch_x), batch_y)
#                 addmse = criterion3(model(batch_x),batch_x - res_band)
#                 epoch_grad += gra.item()
#                 epoch_mse += msel.item()
#                 epoch_add += addmse.item()

                epoch_loss += loss.item()
                epoch_band1 += band1.item()
                epoch_band2 += band2.item()
                epoch_band3 += band3.item()
                epoch_band4 += band4.item()
                epoch_band5 += band5.item()
                epoch_band6 += band6.item()

                loss.backward()
                optimizer.step()
                if n_count % 100 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
        
        
        with torch.no_grad(): # check performance on test dataset, this does not influence the training process
            for tn_count, tbatch_yx in enumerate(testDLoader):
                if cuda:
                    tbatch_x, tbatch_y1 = tbatch_yx[0].to(cuda1), tbatch_yx[1].to(cuda1)
                    model.eval()  
                    
                    tbatch_y = tbatch_y1
                    new_tband6 =  tbatch_x-torch.sum(model(tbatch_x),1,keepdim=True)
                    tres_band = tbatch_x - torch.sum(tbatch_y,1,keepdim=True)

                    test_loss1 = criterion1(model(tbatch_x), tbatch_y)# + criterion2(model(tbatch_x), tbatch_y) + criterion3(model(tbatch_x),tbatch_x - tres_band)
                    test_epoch_loss += test_loss1.item()
                    
#                     tmsel = criterion1(model(tbatch_x), tbatch_y)
#                     tgra = criterion2(model(tbatch_x), tbatch_y)
#                     taddmse = criterion3(model(tbatch_x),tbatch_x - tres_band)
#                     tepoch_add += taddmse.item()
#                     tepoch_grad += tgra.item()
#                     tepoch_mse += tmsel.item()
            
        if epoch % 500 == 0: # save the weights every 500 epochs
            torch.save({'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},os.path.join(save_dir, 'model_%04d_save.pth' % epoch))

        #-----------------------------------------------------------------------------------------------------------------
        # Summary for visualisation in TensorboardX:
        #-----------------------------------------------------------------------------------------------------------------
        elapsed_time = time.time() - start_time
        
        writer.add_scalars('Epoch Losses', {'training_loss': (epoch_loss/(n_count+1)),
                                            'testing_loss': (test_epoch_loss/(tn_count+1))}, epoch+1)
        writer.add_scalars('Band losses', {'Band_1': (epoch_band1/(n_count+1)),
                                           'Band_2': (epoch_band2/(n_count+1)),
                                           'Band_3': (epoch_band3/(n_count+1)),
                                           'Band_4': (epoch_band4/(n_count+1)),
                                           'Band_5': (epoch_band5/(n_count+1)),
                                           'Band_res': (epoch_band6/(n_count+1))},epoch+1)
#         writer.add_scalars('Separated Losses', {'MSE_normal': (epoch_mse/(n_count+1)),
#                                                 'MSE_add': (epoch_add/(n_count+1)),
#                                                 'Huber_grad': (epoch_grad/(n_count+1)),
#                                                 'Testing_MSE_add': (tepoch_add/(tn_count+1)),
#                                                 'Testing_MSE_normal': (tepoch_mse/(tn_count+1)),
#                                                 'Testing_grad': (tepoch_grad/(tn_count+1))},epoch+1)

        writer.add_image('Test 1Band', (model(tbatch_x)[0,0,:,:]-torch.min(model(tbatch_x)[0,0,:,:]))/torch.max(model(tbatch_x)[0,0,:,:]-torch.min(model(tbatch_x)[0,0,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 1GT',(tbatch_y[0,0,:,:]-torch.min(tbatch_y[0,0,:,:]))/torch.max(tbatch_y[0,0,:,:]-torch.min(tbatch_y[0,0,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 2Band', (model(tbatch_x)[0,1,:,:]-torch.min(model(tbatch_x)[0,1,:,:]))/torch.max(model(tbatch_x)[0,1,:,:]-torch.min(model(tbatch_x)[0,1,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 2GT',(tbatch_y[0,1,:,:]-torch.min(tbatch_y[0,1,:,:]))/torch.max(tbatch_y[0,1,:,:]-torch.min(tbatch_y[0,1,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 3Band', (model(tbatch_x)[0,2,:,:]-torch.min(model(tbatch_x)[0,2,:,:]))/torch.max(model(tbatch_x)[0,2,:,:]-torch.min(model(tbatch_x)[0,2,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 3GT',(tbatch_y[0,2,:,:]-torch.min(tbatch_y[0,2,:,:]))/torch.max(tbatch_y[0,2,:,:]-torch.min(tbatch_y[0,2,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 4Band', (model(tbatch_x)[0,3,:,:]-torch.min(model(tbatch_x)[0,3,:,:]))/torch.max(model(tbatch_x)[0,3,:,:]-torch.min(model(tbatch_x)[0,3,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 4GT',(tbatch_y[0,3,:,:]-torch.min(tbatch_y[0,3,:,:]))/torch.max(tbatch_y[0,3,:,:]-torch.min(tbatch_y[0,3,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 5Band', (model(tbatch_x)[0,4,:,:]-torch.min(model(tbatch_x)[0,4,:,:]))/torch.max(model(tbatch_x)[0,4,:,:]-torch.min(model(tbatch_x)[0,4,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 5GT',(tbatch_y[0,4,:,:]-torch.min(tbatch_y[0,4,:,:]))/torch.max(tbatch_y[0,4,:,:]-torch.min(tbatch_y[0,4,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 6Band', (new_tband6[0,0,:,:]-torch.min(new_tband6[0,0,:,:]))/torch.max(new_tband6[0,0,:,:]-torch.min(new_tband6[0,0,:,:])),epoch+1,dataformats='HW')
        writer.add_image('Test 6GT',(tres_band[0,0,:,:]-torch.min(tres_band[0,0,:,:]))/torch.max(tres_band[0,0,:,:]-torch.min(tres_band[0,0,:,:])),epoch+1,dataformats='HW')

        log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/(n_count+1), elapsed_time))
        
        
        
        #-----------------------------------------------------------------------------------------------------------------
        # Save data:
        #-----------------------------------------------------------------------------------------------------------------
        # Save current state_dict of model and optimizer as well as the corresponding epoch for the case in which training interrupts.
        torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},os.path.join(save_dir, 'model_save.pth'))
        
        # Save the different losses
        train_test_loss = [epoch+1,(epoch_loss/(n_count+1)), (test_epoch_loss/(tn_count+1))]
        losses_train_test.append(train_test_loss)
        np.savetxt(os.path.join(save_dir, 'losses_train_test.txt'), losses_train_test,fmt='%.1d %+.18e %+.18e')
        
#         separated_loss = [epoch+1,(epoch_mse/(n_count+1)),(epoch_add/(n_count+1)),(epoch_grad/(n_count+1))] 
#         losses_separated.append(separated_loss)
#         np.savetxt(os.path.join(save_dir, 'losses_separated.txt'), losses_separated,fmt='%.1d %+.18e %+.18e %+.18e')
        
#         test_separated_loss = [epoch+1,(vepoch_mse/(tn_count+1)),(vepoch_add/(tn_count+1)),(vepoch_grad/(tn_count+1))] 
#         test_losses_separated.append(test_separated_loss)
#         np.savetxt(os.path.join(save_dir, 'test_losses_separated.txt'), test_losses_separated,fmt='%.1d %+.18e %+.18e %+.18e')
        
        bands_loss = [epoch+1,(epoch_band1/(n_count+1)), (epoch_band2/(n_count+1)), (epoch_band3/(n_count+1)), (epoch_band4/(n_count+1)), (epoch_band5/(n_count+1)), (epoch_band6/(n_count+1))]
        losses_bands.append(bands_loss)
        np.savetxt(os.path.join(save_dir, 'losses_bands.txt'), losses_bands,fmt='%.1d %+.18e %+.18e %+.18e %+.18e %+.18e %+.18e')
        
    np.savetxt(os.path.join(save_dir, 'elapsed_time.txt'), [time.time() - start_time],fmt='%.18e')
        


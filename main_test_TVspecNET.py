# ---------------------------------------------------------------------------------------------------------------------------
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
# This code is partially taken from the DnCNN implementation from Kai Zhang (2018) https://github.com/cszn/DnCNN and amended for 
# our purpose. 
# The U-Net basis architecture for U-TVspecNET is taken from Naoto Usuyama (2018) https://github.com/usuyama/pytorch-unet/, and 
# the FFDNet basis architecture for F-TVspecNET is taken from Kai Zhang (2019) https://github.com/cszn/KAIR
#
# The following code is to test the TVspecNET model on .mat data type images. If the groundtruth exists, it will automatically 
# evaluate the TV transfrom properties (one-homogeneity, translation and rotation invariance). We include 2 data preprocessing 
# modes for the groundtruth: a) image was decomposed into 50 bands and will he dyadically combined here
#                            b) image was decomposed into 6 dyadic bands
# We additionally enable testing of F-TVspecNET (choose --model 'F-TVspecNET') and U-TVspecNET (choose --model 'U-TVspecNET').
#
# PyTorch Version 1.1.0 used for this implementation. 
# ----------------------------------------------------------------------------------------------------------------------------

import argparse
import os, time, datetime, glob
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision.models as models
import scipy.io as sio
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave


# ============================================================================================================================
# 1. Get parser arguments
# ============================================================================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='./data/test/', type=str, help='directory of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('models', 'TVspecNET'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory to save test results')
    parser.add_argument('--save_result', default=0, type=int, help='save the evaluation measures, 1 or 0')
    parser.add_argument('--save_images', default=0, type=int, help='save the decomposed images, 1 or 0')
    parser.add_argument('--model', default='TVspecNET', type=str, help='choose a type of model')
    parser.add_argument('--cuda_card', default=0, type=int, help='select the cuda card (if multiple available) to run testing on')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path#+'.tif'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        result = result 
        np.save(path,result)

    
# ============================================================================================================================
# 2. Define the model architectures a) TVspecNET, b) F-TVspecNET, c) U-TVspecNET
# ============================================================================================================================  
    
#-------------------------------------------------------------------------------------------------------------------
# a) TVspecNET (with DnCNN as the basis architecture)
#-------------------------------------------------------------------------------------------------------------------
class TVspecNET(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, band_channels = 5, use_bnorm=True, kernel_size=3): #was 17
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
    def __init__(self, image_channels=1, band_channels=5, kernel_size=3): #5, 24
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
# 3. Define evalutation measures: a) MSE, b) sLMSE
# ============================================================================================================================  
#-------------------------------------------------------------------------------------------------------------------
# a) MSE
#-------------------------------------------------------------------------------------------------------------------
def MSE_bands(input, target):
    return np.mean(np.mean(np.mean((input-target)**2,-2),-1)/(np.mean(np.mean(target**2,-2),-1)+1e-4))

#-------------------------------------------------------------------------------------------------------------------
# b) sLMSE 
#-------------------------------------------------------------------------------------------------------------------
def LMSE(input):
    k = 16
    step_size = np.int(k/2)
    LMSE = 0
    counter = 0
    for i in range(0, input.shape[-2] - k + 1, step_size):
        for j in range(0, input.shape[-1] - k + 1, step_size):
            patch = input[:,i:i+step_size, j:j+step_size]
            LMSE += np.mean(np.mean(patch**2,-2),-1) 
            counter += 1
    return (1/counter)*LMSE

def sLMSE(input, target):
    sLMSE = 1 - np.mean(LMSE(input-target)/LMSE(target))
    return sLMSE

# ============================================================================================================================
# 4. Define evaluation approach where ground truth data is available
# ============================================================================================================================ 
def evaluate_output(input, gt_bands, x): # input: inference bands; gt_bands: ground truth bands
    b = []
    
    # residuals:
    input_res = x.cpu().detach().numpy().astype(np.float32).squeeze()-np.sum(input,0)
    gt_bands_res = x.cpu().detach().numpy().astype(np.float32).squeeze()-np.sum(gt_bands,0)    
    # sums over bands:
    input_sum = np.sum(input,0)
    gt_bands_sum = np.sum(gt_bands,0)

    # evaluate decomposition for each band and compare the sums (ssim, psnr, mse, slmse)
    b.append([compare_ssim(gt_bands[0,:,:],input[0,:,:]), compare_psnr(gt_bands[0,:,:], input[0,:,:], data_range=np.maximum(np.max(gt_bands[0,:,:]),np.max(input[0,:,:]))-np.minimum(np.min(gt_bands[0,:,:]),np.min(input[0,:,:]))), MSE_bands(input[0,:,:],gt_bands[0,:,:]),sLMSE(input[0,:,:][None],gt_bands[0,:,:][None])])
    b.append([compare_ssim(gt_bands[1,:,:],input[1,:,:]), compare_psnr(gt_bands[1,:,:], input[1,:,:], data_range=np.maximum(np.max(gt_bands[1,:,:]),np.max(input[1,:,:]))-np.minimum(np.min(gt_bands[1,:,:]),np.min(input[1,:,:]))), MSE_bands(input[1,:,:],gt_bands[1,:,:]),sLMSE(input[1,:,:][None],gt_bands[1,:,:][None])])
    b.append([compare_ssim(gt_bands[2,:,:],input[2,:,:]), compare_psnr(gt_bands[2,:,:], input[2,:,:], data_range=np.maximum(np.max(gt_bands[2,:,:]),np.max(input[2,:,:]))-np.minimum(np.min(gt_bands[2,:,:]),np.min(input[2,:,:]))), MSE_bands(input[2,:,:],gt_bands[2,:,:]),sLMSE(input[2,:,:][None],gt_bands[2,:,:][None])])
    b.append([compare_ssim(gt_bands[3,:,:],input[3,:,:]), compare_psnr(gt_bands[3,:,:], input[3,:,:], data_range=np.maximum(np.max(gt_bands[3,:,:]),np.max(input[3,:,:]))-np.minimum(np.min(gt_bands[3,:,:]),np.min(input[3,:,:]))), MSE_bands(input[3,:,:],gt_bands[3,:,:]),sLMSE(input[3,:,:][None],gt_bands[3,:,:][None])])
    b.append([compare_ssim(gt_bands[4,:,:],input[4,:,:]), compare_psnr(gt_bands[4,:,:], input[4,:,:], data_range=np.maximum(np.max(gt_bands[4,:,:]),np.max(input[4,:,:]))-np.minimum(np.min(gt_bands[4,:,:]),np.min(input[4,:,:]))), MSE_bands(input[4,:,:],gt_bands[4,:,:]),sLMSE(input[4,:,:][None],gt_bands[4,:,:][None])])
    b.append([compare_ssim(gt_bands_res,input_res), compare_psnr(gt_bands_res, input_res, data_range=np.maximum(np.max(gt_bands_res),np.max(input_res))-np.minimum(np.min(gt_bands_res),np.min(input_res))), MSE_bands(input_res,gt_bands_res),sLMSE(input_res[None],gt_bands_res[None])])
    b.append([compare_ssim(gt_bands_sum,input_sum), compare_psnr(gt_bands_sum, input_sum, data_range=np.maximum(np.max(gt_bands_sum),np.max(input_sum))-np.minimum(np.min(gt_bands_sum),np.min(input_sum))), MSE_bands(input_sum,gt_bands_sum),sLMSE(input_sum[None],gt_bands_sum[None])])
    
    # obtain the mean evaluation measures
    ssim_av = np.mean([b[0][0],b[1][0],b[2][0],b[3][0],b[4][0],b[5][0]])
    psnr_av = np.mean([b[0][1],b[1][1],b[2][1],b[3][1],b[4][1],b[5][1]])
    slmse_av = np.mean([b[0][3],b[1][3],b[2][3],b[3][3],b[4][3],b[5][3]])
    
    # evaluation measures without the residual (for one-homogeneity, since they are just zeros)
    ssim_av5 = np.mean([b[0][0],b[1][0],b[2][0],b[3][0],b[4][0]])
    psnr_av5 = np.mean([b[0][1],b[1][1],b[2][1],b[3][1],b[4][1]])
    slmse_av5 = np.mean([b[0][3],b[1][3],b[2][3],b[3][3],b[4][3]])

    return  ssim_av, psnr_av, slmse_av, b, ssim_av5, psnr_av5, slmse_av5


# ============================================================================================================================
# 5. Main testing
# ============================================================================================================================  
if __name__ == '__main__':

    # select architecture
    args = parse_args()
    if args.model == 'TVspecNET':
        model = TVspecNET()
    elif args.model == 'FTVspecNET':
        model = FTVspecNET()
    elif args.model == 'UTVspecNET':
        model = UTVspecNET()
    else:
        print('Error, model name not existent.')

    # load the model weights from the trained model
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        model = torch.load(os.path.join(args.model_dir, 'model_save.pth'))
        # load weights into new model
        log('Load trained model (attention: model name given did not exist, loading model_save.pth)')
    else:
        checkpoint = torch.load(os.path.join(args.model_dir, args.model_name))
        model.load_state_dict(checkpoint['state_dict'])
        log('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        cuda1 = torch.device('cuda',args.cuda_card)
        model = model.to(cuda1)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    set_names = sorted(glob.glob(args.set_dir + 'test*')) # directories of testing images 
    measures = []
    hom_measures = []
    rot_measures = []
    trans_measures = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []
    b6 = []
    comp_sum = []
    
    timer = []
    begin_time = time.time()
    for i in range(len(set_names)):
        sub_dir_name = 'test%03d' % (i+1)
        if args.save_images:
            if not os.path.exists(os.path.join(args.result_dir, sub_dir_name)): 
                os.mkdir(os.path.join(args.result_dir, sub_dir_name))
        
        #-------------------------------------------------------------------------------------------------------------------
        # Load input image
        #-------------------------------------------------------------------------------------------------------------------
        x = sio.loadmat(os.path.join(set_names[i],'image.mat')) 
        x = x['img'] 
        x = np.array(x, dtype='float32')
        x = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])
        torch.cuda.synchronize()
        x = x.cuda(device=cuda1)
        start_time = time.time()
        
        #-------------------------------------------------------------------------------------------------------------------
        # Inference and save results
        #-------------------------------------------------------------------------------------------------------------------
        y_ = model(x) 
        elapsed_time = time.time() - start_time
        timer.append(elapsed_time)
        y_ = y_.cpu()
        y_ = y_.detach().numpy().astype(np.float32)
        y_ = y_.squeeze()        
        torch.cuda.synchronize()
        y_res = x.cpu().detach().numpy().astype(np.float32).squeeze()-np.sum(y_,0)
        
        if args.save_images:
            save_result(x.cpu(), path=os.path.join(args.result_dir,sub_dir_name,'x'))
            save_result(y_, path=os.path.join(args.result_dir,sub_dir_name,'y_'))
        
        #-------------------------------------------------------------------------------------------------------------------
        # Check if there is ground truth data for comparison and if yes evaluate comparisons incl. TV properties
        #------------------------------------------------------------------------------------------------------------------- 
        if len(glob.glob(set_names[i]+'/b*.mat')) !=0 or len(glob.glob(set_names[i]+'/filtered_image*_band*.mat')) !=0:
            #os.path.join
            # Load data of format a) 50 bands and combine them to 6 dyadic bands
            if len(glob.glob(set_names[i]+'/filtered_image*_band*.mat')) !=0:
                bands_list = sorted(glob.glob(set_names[i] + '/filtered_image*_band*.mat'))
                batch_y = []
                for j in range(1,6):#(1,6) 
                    batch_temp=[]
                    for b in range((2**(j-1)-1),2**(j)-1): 
                        # decomposed bands
                        batch = sio.loadmat(bands_list[b])
                        batch = batch['f_H']
                        batch_temp.append(batch)
                    batch_temp = np.sum(np.array(batch_temp,dtype='float32'),0)
                    batch_y.append(batch_temp)
                bands = np.array(batch_y, dtype='float32')

            # Load data of format b) 6 dyadic bands
            elif len(glob.glob(set_names[i]+'/b*.mat')) !=0:
                bands_list = sorted(glob.glob(set_names[i]+'/b*.mat'))
                batch_y = []
                batch = sio.loadmat(bands_list[0])
                batch = np.array(batch['b1'],dtype='float32')
                batch_y.append(batch)
                batch = sio.loadmat(bands_list[1])
                batch = np.array(batch['b2'],dtype='float32')
                batch_y.append(batch)
                batch = sio.loadmat(bands_list[2])
                batch = np.array(batch['b3'],dtype='float32')
                batch_y.append(batch)
                batch = sio.loadmat(bands_list[3])
                batch = np.array(batch['b4'],dtype='float32')
                batch_y.append(batch)
                batch = sio.loadmat(bands_list[4])
                batch = np.array(batch['b5'],dtype='float32')
                batch_y.append(batch)
                bands = np.array(batch_y, dtype='float32')
            
            [ssim_av, psnr_av, slmse_av, b, ssim_av5, psnr_av5, slmse_av5] = evaluate_output(y_, bands, x) # evaluation
            measures.append([i+1,ssim_av,psnr_av,slmse_av])
            b1.append(b[0]) #separate the evaluation results for each band
            b2.append(b[1])
            b3.append(b[2])
            b4.append(b[3])
            b5.append(b[4])
            b6.append(b[5])
            comp_sum.append(b[6])
            
            if args.save_images:
                save_result(bands, path=os.path.join(args.result_dir,sub_dir_name,'GT_bands'))
                np.savetxt(os.path.join(args.result_dir,sub_dir_name, 'compare_measures.txt'), np.hstack((ssim_av,psnr_av,mean_av,slmse_av)))
            
            # 1-hom comparison
            hom_bands = model(2*x)  # inference of multiplied image
            hom_bands = hom_bands.cpu().detach().numpy().astype(np.float32).squeeze()
            torch.cuda.synchronize()
            hom_bands_res = (2*x).cpu().detach().numpy().astype(np.float32).squeeze()-np.sum(hom_bands,0) # residual
            hom_bands2 = hom_bands.copy() # shift bands back according the one-homogeneity property
            hom_bands2[0,:,:] = hom_bands[0,:,:] + hom_bands[1,:,:]
            hom_bands2[1,:,:] = hom_bands[2,:,:]
            hom_bands2[2,:,:] = hom_bands[3,:,:]
            hom_bands2[3,:,:] = hom_bands[4,:,:]
            hom_bands2[4,:,:] = hom_bands_res
            hom_bands2 = 0.5*hom_bands2 # rescale to have the same contrast as the initial image
            y_2 = y_.copy() 
            y_2[4,:,:] += y_res 
            
            [ssim_av, psnr_av, slmse_av, b, ssim_av5, psnr_av5, slmse_av5] = evaluate_output(hom_bands2, y_2, x)
            hom_measures.append([i+1,ssim_av5,psnr_av5,slmse_av5])
        
            # Rotation comparison
            rot_bands = model(torch.rot90(x,1,[2,3]))  # inference of rotated image
            rot_bands = torch.rot90(rot_bands,3,[2,3]) # rotate back
            rot_bands = rot_bands.cpu().detach().numpy().astype(np.float32).squeeze()
            torch.cuda.synchronize()
            rot_bands_res = x.cpu().detach().numpy().astype(np.float32).squeeze()-np.sum(rot_bands,0) # residual
            
            [ssim_av, psnr_av, slmse_av, b, ssim_av5, psnr_av5, slmse_av5] = evaluate_output(rot_bands, y_, x)
            rot_measures.append([i+1,ssim_av,psnr_av,slmse_av])
            
        
            # Translation comparison
            padded_x1 = x.clone() # zero-pad input image (to avoid translation errors around the border)
            padded_x1[0,0,x.shape[2]-5:x.shape[2],:] = 0 # padded image (not translated! - for comparison)
            padded_bands = model(padded_x1.cuda(device=cuda1)) # inference of padded image
            padded_bands = padded_bands.cpu().detach().numpy().astype(np.float32).squeeze()
            
            padded_x = torch.zeros(x.shape) # zero-pad input image
            padded_x[0,0,5:x.shape[2],:] = x[0,0,0:x.shape[2]-5,:] # translate padded image
            padded_x = padded_x.cuda(device=cuda1)
            trans_bands = model(padded_x)  # inference of translated, padded image
            trans_bands = trans_bands.cpu().detach().numpy().astype(np.float32).squeeze()
            torch.cuda.synchronize()
            trans_bands_res = x.cpu().detach().numpy().astype(np.float32).squeeze()-np.sum(trans_bands,0) # residual
        
            bands_tr = np.zeros(trans_bands.shape)
            bands_tr[:,0:x.shape[2]-5,:] = trans_bands[:,5:x.shape[2],:] # translate the decomposed bands back
            bands_tr = np.array(bands_tr,dtype='float32')
            trans_bands_res[0:x.shape[2]-5,:] = trans_bands_res[5:x.shape[2],:]
            trans_bands_res[x.shape[2]-5:x.shape[2],:] = 0
            
            [ssim_av, psnr_av, slmse_av, b, ssim_av5, psnr_av5, slmse_av5] = evaluate_output(bands_tr[:,0:x.shape[2]-5,:], padded_bands[:,0:x.shape[2]-5,:], padded_x1[:,:,0:x.shape[2]-5,:])
            trans_measures.append([i+1,ssim_av,psnr_av,slmse_av])  
            
            if args.save_images:
                save_result(hom_bands, path=os.path.join(args.result_dir,sub_dir_name,'y_1hom'))
                save_result(rot_bands, path=os.path.join(args.result_dir,sub_dir_name,'y_rot'))
                save_result(trans_bands, path=os.path.join(args.result_dir,sub_dir_name,'y_trans'))
                save_result(padded_bands, path=os.path.join(args.result_dir,sub_dir_name,'y_padded'))
    
    #-------------------------------------------------------------------------------------------------------------------
    # Summary of inference run time and evalutation measures (where applicable)
    #------------------------------------------------------------------------------------------------------------------- 
    total_time = (time.time()-begin_time)
    print('Elapsed time: %2.4f second, ' % total_time, 'average inference time per image: %2.4f second.' % (np.mean(timer)))
    
    if measures: # check if measures is non-empty
        # display mean evaluation measures for whole dataset
        measures = np.array(measures, dtype='float32')
        print('Dataset measure comparison: SSIM:', np.mean(measures[:,1],0), ', PSNR:', np.mean(measures[:,2],0), ', sLMSE:', np.mean(measures[:,3],0))
    
        # take mean for each measure of each individual band and display results
        b1 = np.mean(np.array(b1, dtype='float32'),0)
        b2 = np.mean(np.array(b2, dtype='float32'),0)
        b3 = np.mean(np.array(b3, dtype='float32'),0)
        b4 = np.mean(np.array(b4, dtype='float32'),0)
        b5 = np.mean(np.array(b5, dtype='float32'),0)
        b6 = np.mean(np.array(b6, dtype='float32'),0)
        comp_sum = np.mean(np.array(comp_sum, dtype='float32'),0)
        print('Mean measures per band. Band 1:', b1, ', Band 2:', b2, ', Band 3:', b3, ', Band 4:', b4, ', Band 5:', b5, ', Band 6 (res):', b6, ', Compare Sum', comp_sum)
    
        # display mean evaluation measures for TV properties (one-homogeneity, rotational and translational invariance)
        hom_measures = np.array(hom_measures, dtype='float32')
        rot_measures = np.array(rot_measures, dtype='float32')
        trans_measures = np.array(trans_measures, dtype='float32')
        print('1-homogeneity comparison: SSIM:', np.mean(hom_measures[:,1],0), ', PSNR:', np.mean(hom_measures[:,2],0), ', sLMSE:', np.mean(hom_measures[:,3],0))
        print('Rotation comparison: SSIM:', np.mean(rot_measures[:,1],0), ', PSNR:', np.mean(rot_measures[:,2],0), ', sLMSE:', np.mean(rot_measures[:,3],0))
        print('Translation comparison: SSIM:', np.mean(trans_measures[:,1],0), ', PSNR:', np.mean(trans_measures[:,2],0), ', sLMSE:', np.mean(trans_measures[:,3],0))
    
        # save the mean evaluation measure results
        if args.save_result:
            np.savetxt(os.path.join(args.result_dir, 'dataset_measures.txt'), measures) 
            np.savetxt(os.path.join(args.result_dir, 'dataset_measures_mean.txt'), [np.mean(measures[:,1],0),np.mean(measures[:,2],0),np.mean(measures[:,3],0)])
            np.savetxt(os.path.join(args.result_dir, 'one-hom_measures_mean.txt'), [np.mean(hom_measures[:,1],0),np.mean(hom_measures[:,2],0),np.mean(hom_measures[:,3],0)]) 
            np.savetxt(os.path.join(args.result_dir, 'rotation_measures_mean.txt'), [np.mean(rot_measures[:,1],0),np.mean(rot_measures[:,2],0),np.mean(rot_measures[:,3],0)]) 
            np.savetxt(os.path.join(args.result_dir, 'translation_measures_mean.txt'), [np.mean(trans_measures[:,1],0),np.mean(trans_measures[:,2],0),np.mean(trans_measures[:,3],0)]) 
            np.savetxt(os.path.join(args.result_dir, 'elapsed_time.txt'), [total_time,np.mean(timer)])
            np.savetxt(os.path.join(args.result_dir, 'bands_measures.txt'), np.array([b1,b2,b3,b4,b5,b6]))







import torch
import torch.nn as nn
import numpy as np 
import pdb
import cv2
import time
from torch.autograd import Variable
import torch.nn.functional as F
# Flow
# X : 2 x Ho x Wo x b


# h : 3 x Ho x Wo x b

class estimate_light(nn.Module):


    def __init__(self):
        super(estimate_light, self).__init__()
        

    def forward(self, input_img,depth_target,albedo):
        with torch.no_grad():
            depth_target = (depth_target+1.0)/2.0
            input_img_scale = (input_img+1.0)/2.0
            mask = input_img_scale[:,0,:,:].clone()
            mask = mask.unsqueeze(1)
            #mask = ((mask+1)/2.0)*255.0
            input_img = (input_img+1)/2.0
            

            npix=torch.zeros(input_img.shape[0],1)
            for i in range(input_img.shape[0]):
                npix[i,0] = torch.sum(mask[i,:,:]>=0)
            mask_index = (mask>0)
            mask[mask==0]=0
            mask[mask!=0]=1
            input_img = input_img*mask
            mask_im = mask.cpu().data.numpy()
            # print 'mask shape',mask.shape
            # import cv2
            # cv2.imwrite('mask_xx.png',mask_im[0,:,:]*255)
            nrows = input_img.shape[2]
            ncols = input_img.shape[3]
            nchannels = input_img.shape[1]
            

            

            gx, gy = np.linspace(0, nrows-1, nrows), np.linspace(0, nrows-1, nrows)
            grid_x_y = np.stack(np.meshgrid(gx, gy))

            grid_x_y = Variable(torch.FloatTensor(grid_x_y)).cuda()
            xx = grid_x_y[0,:,:]
            yy = grid_x_y[1,:,:]
                
            xx = (xx - 256.0)/608.3650
            yy = (yy - 256.0)/608.3650
            
            xx = xx*depth_target
            yy = yy*depth_target  
            zz = depth_target

            p = torch.zeros(depth_target.shape[0],nrows,ncols,3)
            
            p[:,:,:,0]=xx[:,0,:,:]
            p[:,:,:,1]=yy[:,0,:,:]
            p[:,:,:,2]=zz[:,0,:,:]

            p_ctr = p[:, 1:-1, 1:-1, :]
            vw = p_ctr - p[:, 1:-1, 2:, :]
            vs = p[:, 2:, 1:-1, :] - p_ctr
            ve = p_ctr - p[:, 1:-1, :-2, :]
            vn = p[:, :-2, 1:-1, :] - p_ctr

            normal_1 = torch.cross(vw,vs)
            normal_2 = torch.cross(ve,vn) 

            normal = normal_1+normal_2
            normal_mag = torch.sqrt(pow(normal[:,:,:,0],2)+pow(normal[:,:,:,1],2)+pow(normal[:,:,:,2],2))

            normal_x = normal[:,:,:,0]/normal_mag
            normal_y = normal[:,:,:,1]/normal_mag
            normal_z = normal[:,:,:,2]/normal_mag
            normal_x_pad = torch.zeros(input_img.shape[0],nrows,ncols)
            normal_y_pad = torch.zeros(input_img.shape[0],nrows,ncols)
            normal_z_pad = torch.zeros(input_img.shape[0],nrows,ncols)
            normal_x_pad[:,1:-1,1:-1]=normal[:,:,:,0]
            normal_y_pad[:,1:-1,1:-1]=normal[:,:,:,1]
            normal_z_pad[:,1:-1,1:-1]=normal[:,:,:,2]
            n_x = torch.zeros(input_img.shape[0],nrows,ncols)
            n_y = torch.zeros(input_img.shape[0],nrows,ncols)
            n_z = torch.zeros(input_img.shape[0],nrows,ncols)
            n_mag = torch.zeros(input_img.shape[0],nrows,ncols)
            
            n_x[:,1:-1,1:-1]=normal_x
            n_y[:,1:-1,1:-1]=normal_y
            n_z[:,1:-1,1:-1]=normal_z
            
            n_mag[:,1:-1,1:-1]=normal_mag
            
            x2 = n_x*n_x
            y2 = n_y*n_y
            z2 = n_z*n_z
            xy = n_x*n_y
            xz = n_x*n_z
            yz = n_y*n_z
           
            N = torch.zeros(input_img.shape[0],ncols,nrows,9)
            N=N.cuda()
            pi_pi = 3.14159
            N[:,:,:,0] = pi_pi*torch.Tensor([1/np.sqrt(4*pi_pi)])*n_mag
            N[:,:,:,1] = (2*pi_pi/3.0)*torch.Tensor([np.sqrt(3/(4*pi_pi))])*normal_z_pad
            N[:,:,:,2] = (2*pi_pi/3.0)*torch.Tensor([np.sqrt(3/(4*pi_pi))])*normal_x_pad
            N[:,:,:,3] = (2*pi_pi/3.0)*torch.Tensor([np.sqrt(3/(4*pi_pi))])*normal_y_pad
            N[:,:,:,4] = (pi_pi/4.0)*(0.5)*torch.Tensor([np.sqrt(5/(4*pi_pi))])*((2*z2-x2-y2)*n_mag)
            N[:,:,:,5] = (pi_pi/4.0)*(3*torch.Tensor([np.sqrt(5/(12*pi_pi))]))*(xz*n_mag)
            N[:,:,:,6] = (pi_pi/4.0)*(3*torch.Tensor([np.sqrt(5/(12*pi_pi))]))*(yz*n_mag)
            N[:,:,:,7] = (pi_pi/4.0)*(3*torch.Tensor([np.sqrt(5/(48*pi_pi))]))*((x2-y2)*n_mag)
            N[:,:,:,8] = (pi_pi/4.0)*(3*torch.Tensor([np.sqrt(5/(12*pi_pi))]))*(xy*n_mag)
            
            
            S=torch.zeros(input_img.shape[0],3,9).cuda()
            
            rhoc = albedo
            rhoc = rhoc.cuda()
            rhoc_mask = rhoc*mask
            

           
            input_img_fla = torch.zeros(input_img.shape[0],nchannels,nrows*ncols)
            
            input_img_fla = input_img_fla.cuda()
            
            for i_bat in range(input_img.shape[0]):
                for i_chan in range(nchannels):
                    input_img_fla[i_bat,i_chan,:] = input_img[i_bat,i_chan,:,:].view(-1)
            
            
            N_flatten = torch.zeros(input_img.shape[0],nrows*ncols,9)
            rhoc_N = torch.zeros_like(N_flatten).cuda()
           
            rhoc_mask_fla = rhoc_mask.view(input_img.shape[0],3,-1)
            N_fla = N.view(input_img.shape[0],-1,9)
            for i_n_bat in range(nchannels):
                rhoc_mask_fla_single = rhoc_mask_fla[:,i_n_bat,:]
                
                for j_n_bat in range(9):
                    
                    rhoc_N[:,:,j_n_bat] = N_fla[:,:,j_n_bat].view(input_img.shape[0],-1).cuda()*rhoc_mask_fla_single

            

            
            for i_batfi in range(input_img.shape[0]):
                for i_c in range(3):
                   
                        b = input_img_fla[i_batfi,i_c,:].unsqueeze(1)
                        A = rhoc_N[i_batfi,:,:].squeeze(0)
                        mask_select = mask[i_batfi,:,:]
                        
                        mask_select_fla = mask_select.view(-1)
                        mask_select_ind = mask_select_fla.nonzero()
                        
                        A_nonzero = A[mask_select_ind[:,0],:]
                        b_nonzero = b[mask_select_ind[:,0],:]
                        X,_ = torch.gels(b_nonzero,A_nonzero)
                        X = X.cuda()
                        
                        S[i_batfi,i_c,:]= X[0:9,:].squeeze(1)


            return S
        

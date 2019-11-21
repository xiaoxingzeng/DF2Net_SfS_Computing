import torch
import torch.nn as nn
import numpy as np 
import pdb
import cv2
import time
from torch.autograd import Variable
from Estimate_light import estimate_light
# Flow
# X : 2 x Ho x Wo x b


# h : 3 x Ho x Wo x b

class Depth_to_irra_image(nn.Module):


    def __init__(self):
        super(Depth_to_irra_image, self).__init__()
        self.estimate_light = estimate_light()

    def forward(self, input_img,depth_refine,depth_target,albedo):
        
        light_coef = self.estimate_light(input_img,depth_target,albedo) # [B,3,9]
        
        #light_coef = torch.zeros(input_img.shape[0],3,9).cuda()
        
        depth_refine_scale = (depth_refine+1)/2.0
        depth_target_scale = (depth_target+1)/2.0
        #mask = depth_target_scale.detach()*255
        input_img_scale = (input_img+1)/2.0
#        import cv2
#        output_im = input_img_scale.cpu().data.numpy()
#        output_im_s = output_im[0,:,:]
#        output_im_s = output_im_s.transpose(1,2,0)
#        output_im_s = output_im_s[:,:,(1,2,0)]
#        cv2.imwrite('AB_part2_xx23.png',output_im_s*255)
        mask = input_img_scale[:,0,:,:].detach()
        mask[mask==0]=0
        mask[mask!=0]=1
        mask = mask.unsqueeze(1)
        input_img = (input_img+1)/2.0
        input_img = input_img*mask
        #print 'requires_grad input_img',input_img.requires_grad
        nrows = input_img.shape[2]
        ncols = input_img.shape[3]
        nchannels = input_img.shape[1]


        gx, gy = np.linspace(0, nrows-1, nrows), np.linspace(0, nrows-1, nrows)
        grid_x_y = np.stack(np.meshgrid(gx, gy))
        # grid_x_y = torch.from_numpy(grid_x_y).cuda()
        # grid_x_y = grid_x_y.float()
        #with torch.no_grad():
        grid_x_y = Variable(torch.FloatTensor(grid_x_y)).cuda()
        
        xx = grid_x_y[0,:,:]    
        yy = grid_x_y[1,:,:]
                    
        xx = (xx - 256.0)/608.3650
        yy = (yy - 256.0)/608.3650
        
        # xx = xx*depth_refine_scale
        # yy = yy*depth_refine_scale  
        # zz = depth_refine_scale

        # p = torch.zeros(depth_refine_scale.shape[0],nrows,ncols,3)
        #print 'depth_refine_scale requires_grad',depth_refine_scale.requires_grad
        p1 = (xx*depth_refine_scale).squeeze(1)
        p2 = (yy*depth_refine_scale).squeeze(1)
        p3 = depth_refine_scale.squeeze(1)
        p = torch.stack([p1,p2,p3],dim=3)
        #print 'p.requires_grad',p.requires_grad
        # import pdb
        
        # pdb.set_trace()
        # print 'p.shape'
        # p[:,:,:,0]=xx[:,0,:,:]
        # p[:,:,:,1]=yy[:,0,:,:]
        # p[:,:,:,2]=zz[:,0,:,:]

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
        
        
        rhoc = albedo
        rhoc = rhoc.cuda()
        rhoc = rhoc*mask
        #print 'rhoc requires_grad',rhoc.requires_grad
        option_lambda = 1
        option_mu = 100
        option_nu = 1e-5
        AB = torch.zeros(depth_refine_scale.shape[0],ncols*nrows,nchannels,1).cuda()
        AB_image = torch.zeros(depth_refine_scale.shape[0],nchannels,nrows,ncols).cuda()
        A = torch.zeros(depth_refine_scale.shape[0],ncols*nrows,nchannels,2).cuda()
        B = torch.zeros(depth_refine_scale.shape[0],nrows*ncols,nchannels,1).cuda()
        mask_pix_num=0
        # for i_batch in range(input_img.shape[0]):
        #     for ch in range(nchannels):
        #         data_mask = mask[i_batch,:,:]
        #         data_mask_fla = data_mask.view(-1)
        #         num_nonzero_mask = data_mask_fla.nonzero()
                
        #         mask_pix_num = mask_pix_num + len(num_nonzero_mask)
        #         data_I = input_img[i_batch,ch,:,:]
        #         data_I_fla = data_I.view(-1) 
        #         data_rho = rhoc[i_batch,:,:].clone()
                
        #         data_rho_fla = data_rho.view(-1)
        #         data_xx = p1[i_batch,:,:]
        #         data_xx_fla = data_xx.view(-1)*data_mask_fla
        #         data_yy = p2[i_batch,:,:]
        #         data_yy_fla = data_yy.view(-1)*data_mask_fla
                
        #         data_dz = torch.zeros(nrows,ncols).cuda()
        #         data_dz[1:-1,1:-1] = normal_mag[i_batch,:,:]
        #         data_dz_fla = data_dz.view(-1)*data_mask_fla
        #         data_s = light_coef[i_batch,ch,:]
        #         data_N = N[i_batch,:,:,:]
                
        #         data_N_fla = data_N.view(-1,9)*data_mask_fla.unsqueeze(1)
        #         #A1 = data_rho_fla*(608.3650*data_s[0]-data_xx_fla*data_s[2])/data_dz_fla
        #         #A2 = data_rho_fla*(608.3650*data_s[1]-data_yy_fla*data_s[2])/data_dz_fla
        #         #B1 = data_rho_fla*torch.sum(data_s[2:]*data_N_fla[:,2:9],1) -  data_rho_fla*data_s[2]/data_dz_fla
        #         non_ab_image = data_rho_fla*torch.sum(data_s[0:]*data_N_fla[:,0:9],1)
        #         #AB1 = A1*data_N_fla[:,0]*data_dz_fla + A2*data_N_fla[:,1]*data_dz_fla + B1
        #         AB_image[i_batch,ch,:,:] = non_ab_image.view(nrows,ncols)
        
        for ch in range(nchannels):
                data_mask = mask
                
                data_mask_fla = data_mask.view(input_img.shape[0],-1)
                num_nonzero_mask = data_mask_fla.nonzero()
               
                mask_pix_num = mask_pix_num + len(num_nonzero_mask)
                data_I = input_img[:,ch,:,:]
                data_I_fla = data_I.view(input_img.shape[0],-1) 
                data_rho = rhoc.clone()
                
                data_rho_fla = data_rho[:,ch,:,:].view(input_img.shape[0],-1)
                data_xx = p1
                data_xx_fla = data_xx.view(input_img.shape[0],-1)*data_mask_fla
                data_yy = p2
                data_yy_fla = data_yy.view(input_img.shape[0],-1)*data_mask_fla
               
                data_dz = torch.zeros(input_img.shape[0],nrows,ncols).cuda()
                data_dz[:,1:-1,1:-1] = normal_mag
                
                data_dz_fla = data_dz.view(input_img.shape[0],-1)*data_mask_fla
                
                data_s = light_coef[:,ch,:]
                data_N = N
               
                data_N_fla = data_N.view(input_img.shape[0],-1,9)*(data_mask_fla.unsqueeze(2))
                #A1 = data_rho_fla*(608.3650*data_s[0]-data_xx_fla*data_s[2])/data_dz_fla
                #A2 = data_rho_fla*(608.3650*data_s[1]-data_yy_fla*data_s[2])/data_dz_fla
                #B1 = data_rho_fla*torch.sum(data_s[2:]*data_N_fla[:,2:9],1) -  data_rho_fla*data_s[2]/data_dz_fla
                non_ab_image = data_rho_fla*torch.sum(data_s.unsqueeze(1)*data_N_fla,2)
                #AB1 = A1*data_N_fla[:,0]*data_dz_fla + A2*data_N_fla[:,1]*data_dz_fla + B1
                AB_image[:,ch,:,:] = non_ab_image.view(input_img.shape[0],nrows,ncols)
               
                
        
        mask_pix_num_py = torch.ones(input_img.shape[0],1).cuda()
       
        mask_pix_num_py = mask_pix_num_py*mask_pix_num
        
        return AB_image,normal_mag,mask_pix_num_py,mask
        

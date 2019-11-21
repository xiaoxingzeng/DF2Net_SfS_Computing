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

class make_gradient(nn.Module):


    def __init__(self):
        super(make_gradient, self).__init__()
        

    def forward(self, mask):
        batch_size,nrows,ncols = mask.shape # mask shape B H W
        Omega_padded = F.pad(mask,(1,1,1,1))  # Omega_padded shape B H+2 W+2

        Omega = torch.zeros(mask.shape[0],mask.shape[1],mask.shape[2],4)
        Omega[:,:,:,0] = mask*Omega_padded[:,2:,1:-1]
        Omega[:,:,:,1] = mask*Omega_padded[:,0:-2,1:-1]
        Omega[:,:,:,2] = mask*Omega_padded[:,1:-1,2:]
        Omega[:,:,:,3] = mask*Omega_padded[:,1:-1,0:-2]

        #index_matrix_f = torch.zeros(mask.shape[0],nrows*ncols)
        mask_flatten = torch.zeros(mask.shape[0],nrows*ncols)
        index_matrix = torch.zeros(mask.shape[0],nrows,ncols)
        for i in range(mask.shape[0]):
            # mask_flatten[i,:] = mask[i,:,:].flatten('F')
            mask_rota = mask[i,:,:].tranpose(0,1)
            mask_rota_fla = mask_rota.contiguous().view(-1)
            nonzero_index = mask_rota_fla.nonzero()

            pp = range(1,len(nonzero_index)+1)
            pp = np.array(pp)
            pp_t = torch.from_numpy(pp)
            mask_rota_fla[nonzero_index[:,0]]=pp_t.float()

            mask_rota_fla_reshape = mask_rota_fla.view[nrows,ncols]
            mask_rota_fla_reshape_1 = mask_rota_fla_reshape.tranpose(0,1)
            mask_rota_fla_reshape_2 = mask_rota_fla_reshape_1.tranpose(0,1)
            mask_rota_fla_reshape_3 = mask_rota_fla_reshape_2.tranpose(0,1)

            index_matrix[i,:,:] = mask_rota_fla_reshape_3 
            # index = np.where(mask_flatten>0)
            # index = np.array(index)
            # index_matrix_f[i,index]=range(1,index.shape[1]+1)
            # index_matrix_f[i,:]=range(nrows*ncols)
            # index_matrix[i,:,:]= np.reshape(index_matrix_f[i,:],[512,512],order='F')


        for i_a in range(mask.shape[0]):
            Omega_3 = Omega[i_a,:,:,2]
            Omega_3_r = Omega_3.transpose(0,1)
            Omega_3_r_fla = Omega_3_r.contiguous().view(-1)
            nonzero_a = Omega_3_r_fla.nonzero()
            xc = nonzero_a%mask.shape[1]
            yc = nonzero_a//mask.shape[1]

            index_matrix_copy = index_matrix[i_a,:,:]
            index_matrix_f = index_matrix_copy.tranpose(0,1)
            index_matrix_fla = index_matrix_f.contiguous().view(-1)
            
            indices_centre = index_matrix_fla[nonzero_a[:,0]]
            x_r = xc
            y_r = yc+1
            ind_right = y_r*mask.shape[1]+x_r
            indices_right = index_matrix_fla[ind_right[:,0]]
            II = indices_centre
            JJ = indices_right
            KK = torch.ones(indices_centre.shape[0],1)
            II_cat = torch.cat([II,indices_centre],dim=0) # expect shape n*1
            JJ_cat = torch.cat([JJ,indices_centre],dim=0) # expect shape m*1
            KK_cat = torch.cat([KK,-torch.ones(indices_centre.shape[0],1)],dim=0)

            II_JJ = torch.cat([II_cat.tranpose(1,0),JJ_cat.tranpose(1,0)],dim=0)
            KK_cat_trans = KK_cat.transpose(1,0)
            Dvp = torch.sparse.FloatTensor(II_JJ, KK_cat_trans.float(), torch.Size([nrows*ncols,nrows*ncols]))
            Svp = torch.eye(ncols*nrows) #   need 335 G ram, ram explosion


        cols = (ind // array_shape[0])
        rows = (ind % array_shape[0])
        return (rows, cols)


        return S
        

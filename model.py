##########################################################################
# @File: model.py
# @Author: Zhehan Liang
# @Date: 6/8/2020
# @Intro: GCN/CNEN model network
##########################################################################

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    '''
    GCN model: Z = AXW
    '''
    def __init__(self, dim_in, dim_out):
        super(GCN,self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2, dim_out,bias=False)

    def forward(self, A_s, X_s, A_t, X_t):
        '''
        Calculate three-layer GCN
        '''
        X_s = F.relu(self.fc1(A_s.mm(X_s)))
        X_s = F.relu(self.fc2(A_s.mm(X_s)))
        X_t = F.relu(self.fc1(A_t.mm(X_t)))
        X_t = F.relu(self.fc2(A_t.mm(X_t)))
        return self.fc3(A_s.mm(X_s)), self.fc3(A_t.mm(X_t))

## Cross-graph convolution
class CNEN(nn.Module):
    '''
    CNEN model
    '''
    def __init__(self, dim_in, dim_out):
        super(CNEN,self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in//2,bias=False) # (Front) first GCN layer
        self.fc2 = nn.Linear(dim_in//2, dim_in//2,bias=False) # (Front) second GCN layer
        self.fc3 = nn.Linear(dim_in//2, dim_in//2,bias=False) # Affinity Metric computing layer
        self.fc4 = nn.Linear(dim_in//2, dim_out,bias=False) # Cross-graph GCN layer

    def forward(self, A_s, X_s, A_t, X_t):
        X_s = F.relu(self.fc1(A_s.mm(X_s)))
        X_s = F.relu(self.fc2(A_s.mm(X_s)))
        X_t = F.relu(self.fc1(A_t.mm(X_t)))
        X_t = F.relu(self.fc2(A_t.mm(X_t)))
        S = (self.fc3(X_s).mm(X_t.t())).exp()
        return self.fc4(S.mm(X_s)), self.fc4(S.mm(X_t)), S

def build_model(params):
    '''
    Build model
    '''
    if params.model=='GCN':
        model = GCN(params.init_dim, params.emb_dim)
    else:
        model = CNEN(params.init_dim, params.emb_dim)
    # Initialization cuda
    if params.cuda:
        model.cuda()
    return model

def save_best_model(params, model):
    """
    Save the best model
    """
    if not os.path.exists(params.best_model_path): # If the best model path does not exist, creat it
        os.makedirs(params.best_model_path)
    file_name = params.dataset + str(params.node_num) + '_best_model.pth'
    path = os.path.join(params.best_model_path, file_name)
    print("=====> Saving the best model ...")
    torch.save(model, path)

def reload_best_model(params):
    """
    Reload the best model
    """
    file_name = params.dataset + str(params.node_num) + '_best_model.pth'
    path = os.path.join(params.best_model_path, file_name)
    print("=====> Reloading the best model ...")
    assert os.path.isfile(path) # Check if the best model exists
    model = torch.load(path)
    return model
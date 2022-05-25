import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeRF(nn.Module):
    def __init__(self,input_x = 3, input_d = 3):
        super(NeRF, self).__init__() # init nn.Module
        self.input_x = input_x
        self.input_d = input_d
        self.skip = 4
        D, W = 8, 256
        self.pts_linears = nn.ModuleList()
        for i in range(D):
            if i == 0:
                self.pts_linears.append(nn.Linear(input_x, W)) # device and dtype are default torch.cuda.FloatTensor
            elif i == self.skip:
                self.pts_linears.append(nn.Linear(W + input_x, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))
        self.alpha_linear = nn.Linear(W, 1)
        self.views_linears = nn.ModuleList()
        self.views_linears.append(nn.Linear(W, W))
        self.views_linears.append(nn.Linear(W + input_d, W//2))
        self.views_linears.append(nn.Linear(W//2, 3))
    
    def forward(self, input):
        #input [N, input_x + input_d]
        input_pts, input_views = torch.split(input, [self.input_x, self.input_d], dim = -1)
        out = input_pts
        for i, f in enumerate(self.pts_linears):
            if i == self.skip:
                out = F.relu(f(torch.cat((input_pts,out), -1)))
            else:
                out = F.relu(f(out))
        alpha = F.relu(self.alpha_linear(out)) # official version do relu in raw2outputs function
        
        out = self.views_linears[0](out) # no activation
        out = torch.cat((out,input_views), -1)
        out = F.relu(self.views_linears[1](out))
        out = torch.sigmoid(self.views_linears[2](out)) # official version do it in raw2outputs function
        
        out = torch.cat((out, alpha), -1)
        return out
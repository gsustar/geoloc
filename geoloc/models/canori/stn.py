import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

N_PARAMS = {
    'affine': 6,
    'rotation': 1
}

ACTIVATION_FN = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}

class STNet(nn.Module):
    def __init__(self, embedding_net, reduce_ratio=1.0):
        super().__init__()
        self.embedding_net = embedding_net
        self.stn_mode = embedding_net.stn_mode
        self.reduce_ratio = reduce_ratio

    # def stn(self, x, theta):
    #     rr = self.reduce_ratio
    #     if self.stn_mode == 'affine':
    #         theta1 = theta.view(-1, 2, 3)
    #     else:
    #         cos_t = torch.cos(theta)
    #         sin_t = torch.sin(theta)
    #         zeros = torch.zeros_like(cos_t)
    #         theta1 = torch.stack(
    #             [
    #                 cos_t*rr, -sin_t*rr, zeros,
    #                 sin_t*rr,  cos_t*rr, zeros
    #             ],
    #             dim=1
    #         ).view(-1, 2, 3)

    #     target_size = [
    #         x.size(0), 
    #         x.size(1), 
    #         int(self.reduce_ratio * x.size(2)) , 
    #         int(self.reduce_ratio * x.size(3))
    #     ]
    #     grid = F.affine_grid(theta1, target_size, align_corners=True)
    #     x = F.grid_sample(x, grid, align_corners=True)
    #     return x
    
    def stn(self, x, theta):
        rr = self.reduce_ratio
        if self.stn_mode == 'affine':
            theta1 = theta.view(-1, 2, 3)
        else:
            theta1 = Variable(torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device()), requires_grad=True)
            theta1 = theta1 + 0
            theta1[:, 0, 0] = 1.0
            theta1[:, 1, 1] = 1.0
            if self.stn_mode == 'rotation':
                angle = theta[:, 0]
                theta1[:, 0, 0] = torch.cos(angle) * rr
                theta1[:, 0, 1] = -torch.sin(angle) * rr
                theta1[:, 1, 0] = torch.sin(angle) * rr
                theta1[:, 1, 1] = torch.cos(angle) * rr

        target_size = [
            x.size(0), 
            x.size(1), 
            int(self.reduce_ratio * x.size(2)) , 
            int(self.reduce_ratio * x.size(3))
        ]
        grid = F.affine_grid(theta1, target_size, align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    # def forward_single(self, x):
    #     theta = self.embedding_net(x).view(-1)
    #     v = self.stn(x, theta)
    #     return v, theta
    
    def forward_single(self, x):
        theta = self.embedding_net(x)
        v = self.stn(x, theta)
        return v, theta

    # def forward(self, x0, alpha0, x1, alpha1):
    #     theta0 = self.embedding_net(x0).view(-1)
    #     theta1 = self.embedding_net(x1).view(-1)

    #     v0 = self.stn(x0, theta0)
    #     v1 = self.stn(x0, (alpha1-alpha0+theta1))
    #     return (v0, theta0), (v1, theta1)
    
    def forward(self, x0, alpha0, x1, alpha1):
        theta0 = self.embedding_net(x0)
        theta1 = self.embedding_net(x1)

        v0 = self.stn(x0, theta0)
        v1 = self.stn(x0, (alpha1-alpha0+theta1))
        return (v0, theta0), (v1, theta1)


class EmbeddingNet(nn.Module):
    def __init__(self, backbone, stn_mode="rotation", hidden_dim=4096, activation_fn="tanh", pi_scale=1.0):
        super().__init__()
        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]
        self.backbone = backbone
        self.pi_scale = pi_scale

        in_channels = self.backbone.output_dim
        # TODO: try more parameters here / more depth
        self.fc_loc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_dim, self.stn_n_params),
            ACTIVATION_FN[activation_fn]()
        )

        # Initialize the weights/bias with identity transformation
        if self.stn_mode == 'affine':
            self.fc_loc[-2].weight.data.fill_(0.0)
            self.fc_loc[-2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc_loc[-2].weight.data.fill_(0.0)
            self.fc_loc[-2].bias.data.fill_(0.0)
        else:
            raise ValueError(f"Unknown STN mode: {self.mode}")

    def theta(self, x):
        x = self.backbone(x)
        theta = self.fc_loc(x)
        theta = theta * math.pi * self.pi_scale
        return theta

    def forward(self, x):
        return self.theta(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint):
    """
    Performs Farthest Point Sampling (FPS) to choose clean representative centers.
    Input: [B, N, C] -> Output: [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        # 1. Fresh memory allocation via clone to prevent stride collision
        centroid = xyz[batch_indices, farthest, :].clone().reshape(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        # 2. MPS-SAFE NON-INPLACE MASKING
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Groups neighboring points within a specified spherical radius context.
    Input: [B, N, C], [B, S, C] -> Output: [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    sqdist = square_distance(new_xyz, xyz)
    
    # MPS-SAFE MASKING: Avoid inline modifications
    mask = sqdist > (radius ** 2)
    group_idx = torch.where(mask, torch.tensor(N, device=device), group_idx)
    
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].clone().view(B, S, 1).repeat(1, 1, nsample)
    
    mask_empty = group_idx == N
    group_idx = torch.where(mask_empty, group_first, group_idx)
    
    return group_idx

def square_distance(src, dst):
    """
    Calculates Squared Distance between two sets of points.
    Input: [B, N, C], [B, M, C] -> Output: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class SampleGroupSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(SampleGroupSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        device = xyz.device
        B, N, C = xyz.shape
        
        # 1. Coordinate Downsampling via Farthest Point Sampling (FPS)
        fps_idx = farthest_point_sample(xyz, self.npoint)
        # FORCE SEPARATE MEMORY BLOCK VIA CLONE FOR SLICED TENSORS
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), fps_idx, :].clone()
        
        # 2. Spherical Ball Grouping
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        
        # 🌟 CRITICAL FOR MPS BACKWARD: Slicing by indices yields broken strides. 
        # Forcing clean memory contiguous layouts right after the index gathering.
        grouped_xyz = xyz[torch.arange(B).view(B, 1, 1), idx, :].clone()
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)

        # 3. Aggregating spatial features with structural features
        if points is not None:
            grouped_points = points[torch.arange(B).view(B, 1, 1), idx, :].clone()
            grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        # 4. Local Convolutional Mapping Layer
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # 5. Symmetric Pooling Function
        new_points = torch.max(grouped_points, -1)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class CdPredictorPointNet(nn.Module):
    def __init__(self, num_points=2048):
        super(CdPredictorPointNet, self).__init__()
        
        sa1_npoint = max(1, num_points // 4)
        
        self.sa1 = SampleGroupSetAbstraction(npoint=sa1_npoint, radius=0.1, nsample=32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = SampleGroupSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = SampleGroupSetAbstraction(npoint=1, radius=1.0, nsample=128, in_channel=256, mlp=[256, 512, 1024])
        
        self.regressor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, _, _ = x.shape
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.contiguous().view(B, 1024)
        x = self.regressor(x)
        return x
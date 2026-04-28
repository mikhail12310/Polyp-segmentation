import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs are raw logits, apply sigmoid for probability
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1.0 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss

class DarknessWeightedLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(DarknessWeightedLoss, self).__init__()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, inputs, targets, darkness_maps):
        # Unreduced BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate weight: 1.0 + alpha * darkness_map
        weight = 1.0 + self.alpha * darkness_maps
        
        # Apply weight
        weighted_bce = (bce_loss * weight).mean()
        
        dice_loss = self.dice(inputs, targets)
        return weighted_bce + dice_loss, weighted_bce, dice_loss

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_tv + w_tv

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(UncertaintyWeightedLoss, self).__init__()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, inputs, targets, uncertainty_map):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        weight = 1.0 + self.alpha * uncertainty_map
        
        weighted_bce = (bce_loss * weight).mean()
        
        dice_loss = self.dice(inputs, targets)
        return weighted_bce + dice_loss, weighted_bce, dice_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1.0 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super(ReconstructionLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, recon, target):
        return self.alpha * self.l1(recon, target) + self.beta * self.ssim(recon, target)

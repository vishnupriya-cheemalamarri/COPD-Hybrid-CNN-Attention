# models_hy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------
# Channel Attention (Squeeze-Excitation)
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)           
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

# -------------------------
# Spatial Attention (CBAM spatial)
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # apply channel pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(concat)
        return x * self.sigmoid(out)

# -------------------------
# Channel Attention (CBAM channel)
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# -------------------------
# CBAM module (channel + spatial)
# -------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# -------------------------
# Lightweight Self-Attention (non-local style)
# -------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h*w).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(b, -1, h*w)                       # B x C' x N
        energy = torch.bmm(proj_query, proj_key)                           # B x N x N
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(b, -1, h*w)                   # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))            # B x C x N
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out

# -------------------------
# Hybrid Attention Model
# -------------------------
class HybridAttentionModel(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, num_classes=2, use_cbam=True, use_se=True, use_self_att=True):
        super(HybridAttentionModel, self).__init__()

        # backbone (use torchvision)
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.backbone_out = 512
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            self.backbone_out = 512
        else:
            backbone = models.resnet18(pretrained=pretrained)
            self.backbone_out = 512

        # remove final FC and avgpool replaced
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        self.use_cbam = use_cbam
        self.use_se = use_se
        self.use_self_att = use_self_att

        if use_cbam:
            self.cbam = CBAMBlock(self.backbone_out)
        else:
            self.cbam = None

        if use_se:
            self.se = SEBlock(self.backbone_out)
        else:
            self.se = None

        if use_self_att:
            self.self_att = SelfAttentionBlock(self.backbone_out)
        else:
            self.self_att = None

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x expected to be 3-channel (if grayscale, repeat channels beforehand)
        x = self.feature_extractor(x)  # B x C x H x W

        # Attention modules (order: CBAM -> SE -> Self-attention)
        if self.cbam is not None:
            x = self.cbam(x)

        if self.se is not None:
            x = self.se(x)

        if self.self_att is not None:
            x = self.self_att(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

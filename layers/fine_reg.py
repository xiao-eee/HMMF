import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.utils as KU
import kornia.filters as KF
from layers.encoder import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.LeakyReLU(0.1)):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        
        if res.shape != x.shape:
            res = F.interpolate(res, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        res += x
        return res


class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, dilation=1, norm=None, act=nn.LeakyReLU,bias=False):
        super(Conv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        if act is nn.LeakyReLU:
            model += [act(negative_slope=0.1,inplace=True)]
        elif act is None:
            model +=[]
        else:
            model +=[act()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=True)

class DispEstimator(nn.Module):
    def __init__(self,channel,depth=4,norm=nn.BatchNorm2d,dilation=1):
        super(DispEstimator,self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel,channel,3,act=None,norm=None,dilation=dilation,padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,act=None))
        
        oc = channel
        ic = channel+self.corrks**2
        dilation = 1
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
       
        self.layers = estimator
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1

    def localcorr(self,feat1,feat2):
        feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
        b,c,h,w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1,(13,13),(3,3),border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth,kernel_size=self.corrks,dilation=4,padding=2*(self.corrks-1),stride=1).reshape(b,c,-1,h,w)
        localcorr = (feat2.unsqueeze(2)-feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat,localcorr],dim=1)
        return corr

    def forward(self,feat1,feat2):
        b,c,h,w = feat1.shape
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0,1,0,0] != w-1 or self.scale[0,0,0,0] != h-1:
            self.scale = torch.FloatTensor([w,h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1,feat2)
        for i,layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr,(13,13),(3,3),border_type='replicate')
        disp = corr.clamp(min=-300,max=300)
        
        return disp/self.scale

class DispRefiner(nn.Module):
    def __init__(self,channel,dilation=1,depth=4):
        super(DispRefiner,self).__init__()
        self.preprocessor = nn.Sequential(Conv2d(channel,channel,3,dilation=dilation,padding=dilation,norm=None,act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,norm=None,act=None))
        oc = channel
        ic = channel+2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)
    def forward(self,feat1,feat2,disp):
        
        b=feat1.shape[0]
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b],feat[b:]],dim=1))
        corr = torch.cat([feat,disp],dim=1)
        delta_disp = self.estimator(corr)
        disp = disp+delta_disp
        return disp

class FeaturePyramid(nn.Module):
    def __init__(self, in_channels=64, base_oc=32):   
        super().__init__()
        # 1/1  256×256
        self.conv1 = nn.Sequential(
            Conv2d(in_channels, base_oc,     3, padding=1),   # 64
            ResBlock(Conv2d, base_oc, 3)
        )
        # 1/2 128×128
        self.down2 = nn.Sequential(
            Conv2d(base_oc, base_oc*2, 3, stride=2, padding=1),  # 128
            ResBlock(Conv2d, base_oc*2, 3)
        )
        # 1/4 64×64
        self.down3 = nn.Sequential(
            Conv2d(base_oc*2, base_oc*4, 3, stride=2, padding=1), # 256
            ResBlock(Conv2d, base_oc*4, 3)
        )
    def forward(self, x):   
        feat1 = self.conv1(x)
        
        feat2 = self.down2(feat1)
        
        feat3 = self.down3(feat2)
        
        return feat1, feat2, feat3


class F_RegNet(nn.Module):
    def __init__(self, base_oc=32, matcher_depth=4):   # base_oc 保持 64
        super().__init__()
        self.ir_pyramid   = FeaturePyramid(64, base_oc)
        self.vis_pyramid  = FeaturePyramid(64, base_oc)

        self.matcher1 = DispEstimator(base_oc*2, matcher_depth, dilation=4)  # 128
        self.matcher2 = DispEstimator(base_oc*4, matcher_depth, dilation=2)  # 256
        self.refiner  = DispRefiner(base_oc,   1)                            # 64
        
        self.grid_down = KU.create_meshgrid(128, 128).cuda()
        self.grid_full = KU.create_meshgrid(256, 256).cuda()
        self.scale = torch.FloatTensor([256, 256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
        self.ST = SpatialTransformer(128, 128, True)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # 输出通道为1

    def match(self, ir_feat1, vis_feat1, ir_feat2, vis_feat2, ir_feat3, vis_feat3):
        if self.scale[0, 1, 0, 0] * 2 != ir_feat1.shape[2] - 1 or self.scale[0, 0, 0, 0] * 2 != ir_feat1.shape[3] - 1:
            self.h, self.w = ir_feat1.shape[2], ir_feat1.shape[3]
            self.scale = torch.FloatTensor([self.w, self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(ir_feat1.device)

        disp2_raw = self.matcher2(ir_feat3, vis_feat3)

        disp2 = F.interpolate(disp2_raw, [ir_feat2.shape[2], ir_feat2.shape[3]], mode='bilinear')
        if disp2.shape[2] != self.grid_down.shape[1] or disp2.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(ir_feat2.shape[2], ir_feat2.shape[3]).cuda()

        warped_ir_feat2 = F.grid_sample(ir_feat2, self.grid_down + disp2.permute(0, 2, 3, 1))

        disp1_raw = self.matcher1(warped_ir_feat2, vis_feat2)

        disp1 = F.interpolate(disp1_raw, [ir_feat1.shape[2], ir_feat1.shape[3]], mode='bilinear')
        disp2 = F.interpolate(disp2, [ir_feat1.shape[2], ir_feat1.shape[3]], mode='bilinear')
        if disp1.shape[2] != self.grid_full.shape[1] or disp1.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(ir_feat1.shape[2], ir_feat1.shape[3]).cuda()

        warped_ir_feat1 = F.grid_sample(ir_feat1, self.grid_full + (disp1 + disp2).permute(0, 2, 3, 1))

        disp_scaleup = (disp1 + disp2) * self.scale
        disp = self.refiner(warped_ir_feat1, vis_feat1, disp_scaleup)
        disp = KF.gaussian_blur2d(disp, (17, 17), (5, 5), border_type='replicate') / self.scale
        
        if self.training:
            return disp, disp_scaleup / self.scale, disp2
        return disp, None, None

    def forward(self, ir_feat, vis_feat):
        ir_feat1, ir_feat2, ir_feat3 = self.ir_pyramid(ir_feat)
        vis_feat1, vis_feat2, vis_feat3 = self.vis_pyramid(vis_feat)
        
        disp_12, disp1, disp2 = self.match(
            ir_feat1, vis_feat1,
            ir_feat2, vis_feat2,
            ir_feat3, vis_feat3
        )
        
        disp_12 = F.interpolate(disp_12, [ir_feat.shape[2], ir_feat.shape[3]], mode='bilinear')
        
        reg_ir = self.ST(ir_feat, disp_12)
        reg_ir = self.out(reg_ir)
        
        return reg_ir, disp_12
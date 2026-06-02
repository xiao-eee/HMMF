import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from math import exp
import os
from hyptorch.pmath import dist_matrix,dist
import kornia.filters as KF
os.environ['CUDA_VISIBLE_DEVICES']='0'

"""
hyperbolic optimition learning
"""
class ContrastiveLoss(nn.Module):
    def __init__(self, args, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.hyper_c = args.hyper_c
        self.margin = margin
        
        if self.hyper_c == 0:
            self.dist_f = self.euclidean_dist
        else:
            self.dist_f = self.hyperbolic_dist
    
    def euclidean_dist(self, x, y):
        return (x - y).pow(2).sum(dim=1)
    
    def hyperbolic_dist(self, x, y):
        return dist(x, y, c=self.hyper_c)
    
    def forward(self, x, x_n, x_p):        
        x = x.view(x.size(0), -1)
        x_n = x_n.view(x_n.size(0), -1)
        x_p = x_p.view(x_p.size(0), -1)
        
        distance_positive = self.dist_f(x, x_p)
        distance_negative = self.dist_f(x, x_n)
        
        loss = torch.clamp(distance_positive - distance_negative + self.margin, min=0.0)
        
        return loss.mean()

"""
fusion loss
"""    
class AdvancedReflectionPad(nn.Module):
    def __init__(self, padding):
        super(AdvancedReflectionPad, self).__init__()
        self.padding = padding
    
    def forward(self, x):
        return F.pad(x, [self.padding] * 4, mode='reflect')

class EdgeAwareSobel(nn.Module):
    def __init__(self):
        super(EdgeAwareSobel, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        
        self.border_decay = nn.Parameter(torch.tensor(0.8))
        
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
        self.pad = AdvancedReflectionPad(1)

    def forward(self, x):
        x_pad = self.pad(x)
        sobelx = F.conv2d(x_pad, self.weightx, padding=0)
        sobely = F.conv2d(x_pad, self.weighty, padding=0)
        grad = torch.abs(sobelx) + torch.abs(sobely)
        
        b, c, h, w = grad.shape
        decay_map = torch.ones((b, c, h, w), device=grad.device)
        
        border_size = 3
        for i in range(border_size):
            decay = self.border_decay ** (i+1)
            decay_map[:, :, i, :] *= decay
            decay_map[:, :, -i-1, :] *= decay
            decay_map[:, :, :, i] *= decay
            decay_map[:, :, :, -i-1] *= decay
        
        return grad * decay_map

class FusionSobel_Loss(nn.Module):
    def __init__(self, hyper_intensity=1, hyper_gradient=10, border_size=7):
        super(FusionSobel_Loss, self).__init__()
        self.hyper_intensity = hyper_intensity
        self.hyper_gradient = hyper_gradient
        self.sobel = EdgeAwareSobel()
        self.border_size = border_size
        self.content_mask_cache = {}

    def create_content_mask(self, shape, device):
        key = tuple(shape)
        if key not in self.content_mask_cache:
            b, c, h, w = shape
            
            center_y, center_x = h // 2, w // 2
            
            y, x = torch.meshgrid(
                torch.arange(h, device=device), 
                torch.arange(w, device=device), 
                indexing='ij'
            )
            y, x = y.float(), x.float()
            
            dist = torch.sqrt(
                (x - center_x)**2 + 
                (y - center_y)**2
            )
            
            center_dist = center_x**2 + center_y**2
            max_dist = torch.sqrt(
                torch.tensor(center_dist, device=device, dtype=torch.float)
            )
            
            mask = torch.clamp(1 - (dist / max_dist)**2, 0, 1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            mask = mask.repeat(b, c, 1, 1)
            
            self.content_mask_cache[key] = mask
        return self.content_mask_cache[key]

    def forward(self, image_vis, image_ir, generate_img):
        content_mask = self.create_content_mask(
            generate_img.shape, 
            generate_img.device
        )
        
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(
            x_in_max * content_mask, 
            generate_img * content_mask
        )
        
        y_grad = self.sobel(image_y)
        ir_grad = self.sobel(image_ir)
        generate_img_grad = self.sobel(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        
        loss_grad = F.l1_loss(
            x_grad_joint * content_mask, 
            generate_img_grad * content_mask
        )
        
        loss_total = self.hyper_intensity * loss_in + self.hyper_gradient * loss_grad
        return loss_total

"""
regstration loss
""" 
    
def weightfiledloss(ref, tgt, disp, disp_gt,border_mask = torch.zeros([1, 1, 128, 128])):
        device = ref.device
        border_mask = border_mask.to(device)
        disp_gt = disp_gt.permute(0,3, 1, 2)
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

def l1loss(img1,img2,mask=1,eps=1e-2):
    #img1_resized = torch.nn.functional.interpolate(img1, size=img2.shape[2:], mode='bilinear', align_corners=False)
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,(3,3),(1,1))*mask_
    img2 = KF.gaussian_blur2d(img2,(3,3),(1,1))*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).mean()

def l2loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,(3,3),(1,1))*mask_
    img2 = KF.gaussian_blur2d(img2,(3,3),(1,1))*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
        # self.AP5 = nn.AvgPool2d(5,stride=1,padding=2)
        # self.MP5 = nn.MaxPool2d(5,stride=1,padding=2)

    def forward(self,img1,img2,eps=1e-2):
        #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        mask = mask_.unsqueeze(1)
        mask = mask.permute(0, 2, 1, 3, 4) 
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        #l = l[...,5:-5,10:-10].mean()
        return l

def imgloss(src, tgt,gradientloss,mask=1, weights=[0.1, 0.9]):
        return weights[0] * (l1loss(src, tgt, mask) + 
                             l2loss(src, tgt, mask)) + weights[1] * gradientloss(src, tgt,mask)

def smoothloss(disp,img=None):
    smooth_d=[3*3,7*3,15*3]
    b,c,h,w = disp.shape
    grad = KF.spatial_gradient(disp,order=2).abs().sum(dim=2)[:,:,5:-5,5:-5].clamp(min=1e-9).mean()
    local_smooth_re = 0
    for d in smooth_d:
        local_mean = KF.gaussian_blur2d(disp,(d,d),(d//6,d//6),border_type='replicate')
        local_smooth_re += 1/(d*1.0+1)*(disp-local_mean)[:,:,d//2:-d//2,d//2:-d//2].pow(2).mean()

    return 5000*local_smooth_re + 500*grad

class F_Reg_Loss(nn.Module):
    def __init__(self,args):
        super(F_Reg_Loss,self).__init__()
        self.grad_loss = gradientloss()
        self.hyper_img = args.hyper_img
        self.hyper_ep = args.hyper_ep
        self.hyper_smo = args.hyper_smo
        
    
    def forward(self,ir_reg,ir,flow,flow_gt):
        img_loss = imgloss(ir_reg,ir,self.grad_loss)
        ep_loss= weightfiledloss(ir_reg,ir,flow,flow_gt)
        smo_loss = smoothloss(flow)

        loss = self.hyper_ep*ep_loss+ self.hyper_img*img_loss+self.hyper_smo*smo_loss
        return loss


class C_Reg_Loss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  
        self.beta  = beta   
        self.channel_compress = nn.Conv2d(64, 1, kernel_size=1, bias=False)

    def forward(self, transformed_ir, vis_image, transform_matrix):
        """
        :param transformed_ir: (B,128,H,W)
        :param vis_image:      (B,128,H,W)
        :param transform_matrix: (B,2,3)
        :return: total_loss, grad_loss, ssim_loss, geo_loss
        """
        # 1. 128→1
        ir_1ch  = self.channel_compress(transformed_ir)
        vis_1ch = self.channel_compress(vis_image)

        grad_loss = self.gradient_loss(ir_1ch, vis_1ch)

        ssim_loss = 1 - self.ssim(ir_1ch, vis_1ch)

        geo_loss = self.geometric_consistency_loss(transform_matrix)

        total_loss = self.alpha * (grad_loss + ssim_loss) + self.beta * geo_loss
        return total_loss

    def gradient_loss(self, img1, img2):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)

        grad_x1 = F.conv2d(img1, sobel_x, padding=1)
        grad_y1 = F.conv2d(img1, sobel_y, padding=1)
        grad1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2 + 1e-8)

        grad_x2 = F.conv2d(img2, sobel_x, padding=1)
        grad_y2 = F.conv2d(img2, sobel_y, padding=1)
        grad2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2 + 1e-8)

        return F.l1_loss(grad1, grad2)

    def ssim(self, img1, img2, window_size=11):
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2        = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12   = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def geometric_consistency_loss(self, transform_matrix):
        det = transform_matrix[:, 0, 0] * transform_matrix[:, 1, 1] - \
              transform_matrix[:, 0, 1] * transform_matrix[:, 1, 0]
        det_loss   = F.mse_loss(det, torch.ones_like(det))
        trans_loss = torch.mean(transform_matrix[:, :, 2] ** 2)
        return det_loss + 0.1 * trans_loss




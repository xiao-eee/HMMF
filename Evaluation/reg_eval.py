import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import cv2
import kornia

import math

class NCC(nn.Module):
    """
        Normalized cross correlation.
    """
    def __init__(self):
        super(NCC, self).__init__()

    def similarity_loss(self, tgt, warped_img):
        sizes = np.prod(list(tgt.shape)[1:])
        flatten1 = torch.reshape(tgt, (-1, sizes))
        flatten2 = torch.reshape(warped_img, (-1, sizes))

        mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
        mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
        var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
        var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
        cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
        pearson_r = cov12 / torch.sqrt((var1 + 1e-5) * (var2 + 1e-6))
        # ncc_value = torch.sum(1 - pearson_r)
        ncc_value = torch.sum(pearson_r)
        return ncc_value

    def forward(self, y_true, y_pred):
        return self.similarity_loss(y_true, y_pred)

def MI(im1, im2, gray_level):
	hang, lie = im1.shape
	count = hang * lie
	N = gray_level
	h = np.zeros((N, N))
	for i in range(hang):
		for j in range(lie):
			h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
	h = h / np.sum(h)
	im1_marg = np.sum(h, axis=0)
	im2_marg = np.sum(h, axis=1)
	H_x = 0
	H_y = 0
	for i in range(N):
		if (im1_marg[i] != 0):
			H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
	for i in range(N):
		if (im2_marg[i] != 0):
			H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
	H_xy = 0
	for i in range(N):
		for j in range(N):
			if (h[i, j] != 0):
				H_xy = H_xy + h[i, j] * math.log2(h[i, j])
	MI = H_xy - H_x - H_y
	return MI

def imread(path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):
    im_cv = cv2.imread(str(path), flags)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    return im_ts.unsqueeze(0) if unsqueeze else im_ts

def calc_img_metrics(ncc_metric, mi_metric, root_in, root_gt):
    NCC_list = []
    MI_list = []
    in_img_list = sorted(os.listdir(root_in))
    gt_img_list = sorted(os.listdir(root_gt))

    for in_img, gt_img in zip(in_img_list, gt_img_list):
        in_img_path = os.path.join(root_in, in_img)
        gt_img_path = os.path.join(root_gt, gt_img)

        img_in = imread(in_img_path, unsqueeze=True).cuda()
        img_gt = imread(gt_img_path, unsqueeze=True).cuda()
        
        # Resize images to match dimensions
        if img_in.shape != img_gt.shape:
            img_in = F.interpolate(img_in, size=img_gt.shape[-2:], mode='bilinear', align_corners=False)

        ncc_value  = ncc_metric(img_gt, img_in)
        
        # Convert to 2D numpy arrays with values in 0-255
        img_gt_np = (img_gt.squeeze().cpu().numpy() * 255).astype(np.uint8)
        img_in_np = (img_in.squeeze().cpu().numpy() * 255).astype(np.uint8)
        mi_value = mi_metric(img_gt_np, img_in_np, gray_level=256)

        NCC_list.append(ncc_value.item())
        MI_list.append(mi_value)
        print("{} NCC = {:.5}, MI = {:.5}".format(
            in_img,  ncc_value.item(), mi_value.item()))
    
    log = 'Average NCC={:.5}, MI={:.5}'.format( 
        sum(NCC_list)/len(NCC_list), 
        sum(MI_list)/len(MI_list))
    print(log)

    return NCC_list, MI_list,log


if __name__ == '__main__':
    # TODO: RoadScene
    # root_gt = './dataset/RoadScene/ir'
    # root_in = './RoadScene/reg'
    # # TODO: TNO
    root_gt = './dataset/TNO/ir'
    root_in = './TNO/reg'
    # TODO: M3FD
    # root_gt = './dataset/M3FD/ir'
    # root_in = './M3FD/reg'

    ncc_metric  = NCC().cuda()
    mi_metric = MI
    # TODO: Calculate Mse and NCC metrics
    NCC_list, MI_list, log = calc_img_metrics(ncc_metric, mi_metric, root_in, root_gt)
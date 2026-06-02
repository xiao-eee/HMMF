import os
import cv2
import numpy as np
import torch
import time
import kornia.utils

def RGB2YCrCb(rgb_image):
    """
    return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

def RF_imsave(img, filename):
    img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
    img = img.squeeze().cpu()
    img = kornia.utils.tensor_to_image(img) * 255.0
    cv2.imwrite(filename,img)

if __name__ == '__main__':
    fuse_y_path = "./dataset/TNO/fus"
    vi_path     = "./dataset/TNO/vi"
    # building save path
    rgb_fuse_path = "./dataset/TNO/fus_rgb"
    if not os.path.exists(rgb_fuse_path):
        os.makedirs(rgb_fuse_path)
    # get raw image list
    y_file_list  = sorted(os.listdir(fuse_y_path))
    vi_file_list = sorted(os.listdir(vi_path))

    for idx, (y_filename, vi_filename) in enumerate(zip(y_file_list, vi_file_list)):

        y_filepath  = os.path.join(fuse_y_path, y_filename)
        vi_filepath = os.path.join(vi_path, vi_filename)
        # read Y_fused image & visible image
        fuse_y  = cv2.imread(y_filepath,  flags=cv2.IMREAD_GRAYSCALE)
        img_vi  = cv2.imread(vi_filepath, flags=cv2.IMREAD_COLOR)
        if fuse_y.shape != img_vi[:,:,0].shape:
            fuse_y = cv2.resize(fuse_y, (img_vi[:,:,0].shape[1], img_vi[:,:,0].shape[0]))

        # get cb and cr channels of the visible image
        vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
        vi_y  = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]
        # get BGR-fused image
        fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
        fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
        # save RGB-fused image
        bgr_fuse_save_name = os.path.join(rgb_fuse_path, y_filename)
        cv2.imwrite(bgr_fuse_save_name, fused_bgr)

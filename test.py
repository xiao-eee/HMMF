import sys
import argparse
import pathlib
import warnings
import statistics
import time
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
from torch import Tensor
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from dataloader.joint_data import JointTestData
from layers.encoder import Shared_Encoder,Euclidean_Encoder,Hyperbolic_Encoder
from layers.coar_reg import C_RegNet
from layers.fine_reg import F_RegNet
from layers.fusion import FusionNet
from layers.TAFF import TAFF
from data.fus_Y_rgb import RF_imsave,RGB2YCrCb,YCbCr2RGB


def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RegNet & FuseNet eval process')
    #TNO dataset
    # parser.add_argument('--ir',   default='./dataset/TNO/ir_warp',   type=pathlib.Path)
    # parser.add_argument('--vi',   default='./dataset/TNO/vi',                 type=pathlib.Path)
    # parser.add_argument('--dst_reg',  default='./dataset/TNO/reg/', help='registration image save folder', type=pathlib.Path)
    # parser.add_argument('--dst_fus',  default='./dataset/TNO/fus/',help='fuse image save folder', type=pathlib.Path)

    # M3FD dataset
    # parser.add_argument('--ir', default='./dataset/M3FD/ir_warp', type=pathlib.Path)
    # parser.add_argument('--vi', default='./dataset/M3FD/vi', type=pathlib.Path)
    # parser.add_argument('--dst_reg',  default='./dataset/M3FD/reg/', help='registration image save folder', type=pathlib.Path)
    # parser.add_argument('--dst_fus',  default='./dataset/M3FD/fus/',help='fuse image save folder', type=pathlib.Path)

    # RoadScene dataset
    parser.add_argument('--ir',   default='./dataset/RoadScene/ir_warp',   type=pathlib.Path)
    parser.add_argument('--vi',   default='./dataset/RoadScene/vi',                 type=pathlib.Path)
    parser.add_argument('--dst_reg',  default='./dataset/RoadScene/reg/', help='registration image save folder', type=pathlib.Path)
    parser.add_argument('--dst_fus',  default='./dataset/RoadScene/fus/',help='fuse image save folder', type=pathlib.Path)

    # checkpoint and save path
    parser.add_argument('--ckpt_base', default='./cache/model_test.pth', type=pathlib.Path, help='checkpoint path')

    # setup
    parser.add_argument('--in_channels', default=1, type=int, help='AFuse feather dim')
    parser.add_argument('--out_channels', default=64,type=int, help='AFuse feather dim')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument('--hyper_c', type=float, default=0.3, help='hyperbolic')
    parser.add_argument('--clip_r', type=float, default=2.3, help='clip radius')

    args = parser.parse_args()
    return args

def main(args):
    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True
    # REMOVED: crop = torchvision.transforms.Resize((128,128))

    print("===> Loading datasets")
    # Pass None instead of crop to keep original dimensions
    data = JointTestData(args.ir, args.vi, crop=None)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    shared_encoder = nn.DataParallel(Shared_Encoder(base_ic=args.in_channels,base_oc = args.out_channels)).to(device)
    euclidean_encoder = nn.DataParallel(Euclidean_Encoder(32,32,32).to(device))
    hyperbolic_encoder = nn.DataParallel(Hyperbolic_Encoder(args)).to(device)
    c_regnet = nn.DataParallel(C_RegNet()).to(device)
    taff_f= nn.DataParallel(TAFF(args.out_channels*2, vis=False)).to(device)  # vis=True for visualization
    taff_r= nn.DataParallel(TAFF(args.out_channels*2, vis=False)).to(device)  # vis=True for visualization
    f_regnet = nn.DataParallel(F_RegNet()).to(device)
    fusnet = nn.DataParallel(FusionNet(args.out_channels)).to(device)

    print("===> loading trained model '{}'".format(args.ckpt_base))
    shared_encoder.load_state_dict(torch.load(args.ckpt_base)['Shared_Encoder'])
    euclidean_encoder.load_state_dict(torch.load(args.ckpt_base)['Euclidean_Encoder'])
    hyperbolic_encoder.load_state_dict(torch.load(args.ckpt_base)['Hyperbolic_Encoder'],strict=False)
    taff_f.load_state_dict(torch.load(args.ckpt_base)['TAFF_F'])
    taff_r.load_state_dict(torch.load(args.ckpt_base)['TAFF_R'])
    c_regnet.load_state_dict(torch.load(args.ckpt_base)['C_RegNet'])
    f_regnet.load_state_dict(torch.load(args.ckpt_base)['F_RegNet'])
    fusnet.load_state_dict(torch.load(args.ckpt_base)['FusionNet'])

    print("===> Starting Testing")
    os.makedirs(args.dst_fus, exist_ok=True)
    os.makedirs(args.dst_reg, exist_ok=True)
    test(shared_encoder,hyperbolic_encoder,euclidean_encoder, c_regnet, f_regnet, fusnet, test_data_loader, args.dst_reg, args.dst_fus, device,taff_f,taff_r,args)
    pass

def test(shared_encoder, hyperbolic_encoder, euclidean_encoder, c_regnet, f_regnet, fusnet, 
         test_data_loader, dst_reg, dst_fus, device, taff_f, taff_r,args):
    shared_encoder.eval()
    hyperbolic_encoder.eval()
    euclidean_encoder.eval()
    taff_f.eval()
    taff_r.eval()
    c_regnet.eval()
    f_regnet.eval()
    fusnet.eval()

    full_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)
    for (ir, vi), (ir_path, vi_path) in tqdm_loader:
        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext
        ir = ir.cuda()
        vi = vi.cuda()
        
        B, C, H_orig, W_orig = ir.shape
        
        pad_h = (8 - H_orig % 8) % 8
        pad_w = (8 - W_orig % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            padder = torch.nn.ReflectionPad2d((0, pad_w, 0, pad_h))
            
            ir = padder(ir)
            vi = padder(vi)
            
            _, _, H_pad, W_pad = ir.shape

        # Registration & Fusion
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            ir_feat = shared_encoder(ir)
            vi_feat = shared_encoder(vi)

            c_reg_ir, transform_matrix = c_regnet(ir_feat, vi_feat)
                
            c_reg_ir_E = euclidean_encoder(c_reg_ir)
            vis_E = euclidean_encoder(vi_feat)

            c_reg_ir_mu, c_reg_ir_H = hyperbolic_encoder(c_reg_ir)
            vis_mu, vis_H = hyperbolic_encoder(vi_feat)
            
            H_features = torch.cat((c_reg_ir_H, vis_H), dim=1)
            E_features = torch.cat((c_reg_ir_E, vis_E), dim=1)
            fus_features = taff_f(H_features, E_features)
            reg_features = taff_r(H_features, E_features)

            F_vis = fus_features[:, :args.out_channels, :, :]
            F_ir = fus_features[:, args.out_channels:, :, :]
            R_vis = reg_features[:, :args.out_channels, :, :]
            R_ir = reg_features[:, args.out_channels:, :, :]

            fus = fusnet(F_vis, F_ir)
            f_reg_ir, final_flow = f_regnet(R_ir, R_vis)
        
        end = time.time()
        full_time.append(end - start)

        if pad_h > 0 or pad_w > 0:
            f_reg_ir = f_reg_ir[:, :, :H_orig, :W_orig]
            fus = fus[:, :, :H_orig, :W_orig]
            final_flow = final_flow[:, :, :H_orig, :W_orig]
        
        grid = kornia.utils.create_meshgrid(H_orig, W_orig, device=ir.device).to(ir.dtype)
        grid = grid.permute(0, 3, 1, 2)

        img_grid = _draw_grid(ir[:, :, :H_orig, :W_orig].squeeze().cpu().numpy(), 24)

        new_grid = grid + final_flow
        new_grid = new_grid.permute(0, 2, 3, 1)

        warp_grid = torch.nn.functional.grid_sample(img_grid.unsqueeze(0), new_grid, padding_mode='border', align_corners=False)
        warp_combine = 0.8 * ir[:, :, :H_orig, :W_orig] + 0.2 * warp_grid
        warp_combine = torch.clamp(warp_combine, 0, 1)

        RF_imsave(f_reg_ir, dst_reg / file_name)
        imsave(img_grid, dst_reg / 'grid', file_name)
        imsave(warp_grid, dst_reg / 'warp_grid', file_name)
        imsave(warp_combine, dst_reg / 'ir_warp_grid', file_name)
        save_flow(final_flow, dst_reg / 'ir_flow', file_name)
        RF_imsave(fus, dst_fus / file_name)
    
def _draw_grid(im_cv, grid_size: int = 24, height: int = None, width: int = None):
    # Use provided dimensions or infer from input
    if height is None or width is None:
        height, width = im_cv.shape
    else:
        # Create a blank image with specified dimensions
        im_cv = np.zeros((height, width), dtype=np.float32)
    
    im_gd_cv = np.full_like(im_cv, 255.0)
    im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)

    color = (0, 0, 255)
    for x in range(0, width - 1, grid_size):
        cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1)
    for y in range(0, height - 1, grid_size):
        cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)
    im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
    return im_gd_ts

def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(p), im_cv)

def save_flow(flow: [Tensor], dst: pathlib.Path, im_name: str = ''):
    rgb_flow = flow2rgb(flow, max_value=None) # (3, 512, 512) type; numpy.ndarray
    im_s = rgb_flow if type(rgb_flow) == list else [rgb_flow]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = (im_ts * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(str(p), im_cv)

def flow2rgb(flow_map: [Tensor], max_value: None):
    flow_map_np = flow_map.squeeze().detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0, 1)
    return rgb_flow

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)
import pathlib
import warnings
import logging.config
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms
import visdom
from tqdm import tqdm
import torch.nn as nn
from data.generate_affine_deform import AffineTransform,ElasticTransform
from dataloader.joint_data import JointTrainData
from layers.encoder import Shared_Encoder,Euclidean_Encoder,Hyperbolic_Encoder
from layers.coar_reg import C_RegNet
from layers.fine_reg import F_RegNet
from layers.fusion import FusionNet
from layers.TAFF import TAFF
from loss import C_Reg_Loss,F_Reg_Loss,FusionSobel_Loss,ContrastiveLoss
import argparse

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

def hyper_args():
# Training settings
    parser = argparse.ArgumentParser(description="PyTorch Corss-modality Registration and Fusion")
    # RoadScene dataset
    parser.add_argument('--ir', default='./dataset/train/ir', type=pathlib.Path)
    parser.add_argument('--vi', default='./dataset/train/vis', type=pathlib.Path)
    parser.add_argument('--ir_mask', default='./dataset/train/ir_sam_mask', type=pathlib.Path)
    parser.add_argument('--vi_mask', default='./dataset/train/vi_sam_mask', type=pathlib.Path)

    # train loss weights
    #fusion loss
    parser.add_argument('--hyper_ssim', default=1.0, type=float)
    parser.add_argument('--hyper_ssim_att', default=20.0, type=float)
    parser.add_argument('--hyper_grad', default=5.0, type=float)
    #fine registration loss
    parser.add_argument('--hyper_img', default=10.0, type=float)
    parser.add_argument('--hyper_ep', default=1.0, type=float)
    parser.add_argument('--hyper_smo', default=1.0, type=float)

    # implement details
    parser.add_argument('--in_channels', default=1, type=int, help='AFuse feather dim')
    parser.add_argument('--out_channels', default=64, type=int, help='AFuse feather dim')
    parser.add_argument("--batchsize", type=int, default=8, help="training batch size")
    parser.add_argument("--nEpochs", type=int, default=1200,help="number of epochs to train for")
    parser.add_argument('--hyper_c', type=float, default=0.3, help='hyperbolic')
    parser.add_argument('--clip_r', type=float, default=2.3, help='clip radius')
    #parser.add_argument("--Epoch_gap", type=int, default=0, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
    parser.add_argument("--step", type=int, default=300, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--interval', default=20, help='record interval')
    # save path of model, please replace 'joint_checkpoint_path' with yourself filename
    parser.add_argument("--ckpt", default="./cache/", type=str, help="path to save model (default: none)")

    args = parser.parse_args()
    return args


def main(args, visdom):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch    = args.nEpochs
    #epoch_gap = args.Epoch_gap
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)
    pathlib.Path(cache).mkdir(parents=True, exist_ok=True)

    print("===> Loading datasets")
    # (256, 256) for RoadScene dataset;
    crop = torchvision.transforms.Resize((128, 128))
    data = JointTrainData(args.ir,args.vi, args.ir_mask, args.vi_mask,crop)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building models")
    shared_encoder = nn.DataParallel(Shared_Encoder(base_ic=args.in_channels,base_oc = args.out_channels)).to(device)
    euclidean_encoder = nn.DataParallel(Euclidean_Encoder(32,32,32)).to(device)
    hyperbolic_encoder = nn.DataParallel(Hyperbolic_Encoder(args)).to(device)
    taff_f= nn.DataParallel(TAFF(args.out_channels*2, vis=False)).to(device)  # vis=True for visualization
    taff_r= nn.DataParallel(TAFF(args.out_channels*2, vis=False)).to(device)  # vis=True for visualization
    c_regnet = nn.DataParallel(C_RegNet()).to(device)
    f_regnet = nn.DataParallel(F_RegNet()).to(device)
    fusnet = nn.DataParallel(FusionNet(args.out_channels)).to(device)

    print("===> Defining Loss fuctions")
    c_reg_loss = C_Reg_Loss().to(device)
    fus_loss = FusionSobel_Loss().to(device)
    f_reg_loss = F_Reg_Loss(args).to(device)
    con_loss = ContrastiveLoss(args).to(device)

    print("===> Setting Optimizers")
    all_params =list(c_regnet.parameters())+list(shared_encoder.parameters())+list(hyperbolic_encoder.parameters())+list(f_regnet.parameters())+list(fusnet.parameters())+list(euclidean_encoder.parameters())+list(taff_f.parameters())+list(taff_r.parameters())
    optimizer = torch.optim.Adam(params=all_params, lr=args.lr)

    print("===> Building deformation")
    affine  = AffineTransform(translate=0.01)
    elastic = ElasticTransform(kernel_size=101, sigma=32) # 16

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        shared_encoder.train()
        c_regnet.train()
        euclidean_encoder.train()
        hyperbolic_encoder.train()
        taff_f.train()
        taff_r.train()
        f_regnet.train()
        fusnet.train()

        # TODO: update learning rate of the optimizer
        lr = adjust_learning_rate(args, optimizer, epoch - 1)
        print("Epoch={}, lr={}".format(epoch,lr))

        tqdm_loader = tqdm(training_data_loader, disable=True)
        num = len(training_data_loader)
        step = 0.
        loss_total, loss_c_reg,loss_fus, loss_f_reg,loss_con = [], [],[], [],[]
        for index, ((ir, vi, ir_mask, vi_mask), _) in enumerate(tqdm_loader):
            step += 1.0 / num
            ir, vi = ir.cuda(), vi.cuda()
            ir_mask, vi_mask = ir_mask.cuda(), vi_mask.cuda()

            # TODO: generate warped ir images
            ir_affine, ir_affine_disp   = affine(ir)
            ir_elastic, ir_elastic_disp = elastic(ir_affine)
            disp_ir = ir_affine_disp + ir_elastic_disp  # cumulative disp grid [batch_size, height, weight, 2]
            ir_warp = ir_elastic

            ir_warp.detach_()
            disp_ir.detach_()

            ir_feat = shared_encoder(ir_warp)
            vi_feat = shared_encoder(vi)
            c_reg_ir,transform_matrix = c_regnet(ir_feat, vi_feat)
                
            positive_ir = c_reg_ir * ir_mask
            negative_ir = vi_feat * vi_mask
            positive_vis = vi_feat * (1 - vi_mask)
            negative_vis = c_reg_ir * (1 - ir_mask)

            c_reg_ir_mu, c_reg_ir_H = hyperbolic_encoder(c_reg_ir)
            assert not torch.isnan(c_reg_ir_mu).any(), "c_reg_ir_mu has NaN"
            assert not torch.isnan(c_reg_ir_H).any(), "c_reg_ir_H has NaN"
            vis_mu, vis_H = hyperbolic_encoder(vi_feat)
            assert not torch.isnan(vis_mu).any(), "vis_mu has NaN"
            assert not torch.isnan(vis_H).any(), "vis_H has NaN"

            positive_ir_mu,_ = hyperbolic_encoder(positive_ir)
            negative_ir_mu,_ = hyperbolic_encoder(negative_ir)
            positive_vis_mu,_ = hyperbolic_encoder(positive_vis)
            negative_vis_mu,_ = hyperbolic_encoder(negative_vis)

            c_reg_ir_E = euclidean_encoder(c_reg_ir)
            vis_E = euclidean_encoder(vi_feat) 

            H_features = torch.cat((c_reg_ir_H, vis_H), dim=1)
            E_features = torch.cat((c_reg_ir_E, vis_E), dim=1)
            fus_features = taff_f(H_features, E_features)
            reg_features = taff_r(H_features, E_features)

            F_vis = fus_features[:, :args.out_channels, :, :]
            F_ir = fus_features[:, args.out_channels:, :, :]
            R_vis = reg_features[:, :args.out_channels, :, :]
            R_ir = reg_features[:, args.out_channels:, :, :]

            fus = fusnet(F_vis,F_ir)
            f_reg_ir,final_flow = f_regnet(R_ir,R_vis)

            C_reg_loss = c_reg_loss(c_reg_ir,vi_feat, transform_matrix)
            Con_loss_ir = con_loss(c_reg_ir_mu,negative_ir_mu,positive_ir_mu)
            Con_loss_vi = con_loss(vis_mu,negative_vis_mu,positive_vis_mu)
            Con_loss = Con_loss_ir+Con_loss_vi
            Fus_loss = fus_loss(vi,f_reg_ir,fus)
            F_reg_loss = f_reg_loss(f_reg_ir, ir,final_flow,disp_ir)
            Total_loss = Fus_loss + F_reg_loss+C_reg_loss+Con_loss

            optimizer.zero_grad()
            Total_loss.backward()           
            torch.nn.utils.clip_grad_norm_(all_params, args.clip_r)
            optimizer.step()

            loss_c_reg.append(C_reg_loss.item())
            loss_fus.append(Fus_loss.item())
            loss_f_reg.append(F_reg_loss.item())
            loss_con.append(Con_loss.item())
            loss_total.append(Total_loss.item())

            if index % 2 == 0:
                c_reg_ir = c_reg_ir[0].unsqueeze(1)
                show = torch.stack([vi[0],ir_warp[0],c_reg_ir[0],f_reg_ir[0],fus[0]])
                show = show.clamp(0, 1)
                visdom.images(show, win='HMMF')

        l = len(loss_total)
        loss_c_reg = sum(loss_c_reg) / l if l > 0 else 0
        loss_total = sum(loss_total) / l if l > 0 else 0
        loss_fus = sum(loss_fus) / l if l > 0 else 0
        loss_f_reg = sum(loss_f_reg) / l if l > 0 else 0
        loss_con = sum(loss_con) / l if l > 0 else 0
        dsp = f'epoch: [{epoch}/{args.nEpochs}] loss: {loss_total: .2f}'
        log.info(dsp)
        tqdm_loader.set_description(dsp)
        # TODO: visdom display
        visdom.line([loss_total], [epoch], win='loss-total', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')
        visdom.line([loss_c_reg], [epoch], win='loss-c-reg', name='creg', opts=dict(title='CReg-loss'), update='append' if epoch else '')
        visdom.line([loss_fus], [epoch], win='loss-fus', name='fus', opts=dict(title='Fus-loss'), update='append' if epoch else '')
        visdom.line([loss_f_reg], [epoch], win='loss-f-reg', name='freg', opts=dict(title='FReg-loss'), update='append' if epoch else '')
        visdom.line([loss_con], [epoch], win='loss-con', name='con', opts=dict(title='Con-loss'), update='append' if epoch else '')
        # TODO: save checkpoint
        if epoch % args.interval == 0:
            checkpoint = {
                'Shared_Encoder': shared_encoder.state_dict(),
                'C_RegNet': c_regnet.state_dict(),
                'Euclidean_Encoder': euclidean_encoder.state_dict(),
                'Hyperbolic_Encoder':hyperbolic_encoder.state_dict(),
                'TAFF_F': taff_f.state_dict(),
                'TAFF_R': taff_r.state_dict(),
                'F_RegNet': f_regnet.state_dict(),
                'FusionNet': fusnet.state_dict(),
            }
            model_out_path = str(cache / f'model_{epoch:04d}.pth')
            torch.save(checkpoint, model_out_path)

            
def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    visdom = visdom.Visdom(port=8097, env='HMMF')
    if not visdom.check_connection():
        raise RuntimeError("Visdom server is not running. Please start it with 'python -m visdom.server'")

    main(args, visdom)
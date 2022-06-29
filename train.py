"""The current goal of this file is to re-implement (copy) the part of the code in original 
    tensoRF implemention that corresponding to train the model."""
import os
import torch

from opt import config_parser #TODO: implement opt.py
from utils import *
from dataLoader import dataset_dict
from torch.utils.tensorboard import SummaryWriter
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, 
                            is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train,
                            is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
         logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/img_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgbs', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp = n_lamb_sigma, appearance_n_comp = n_lamb_sh, app_dim = args.data_dim_color, near_far = near_far,
                    shadingMode = args.shadingMode, alphaMask_thre = args.alpha_mask_thre, density_shift = args.density_shift, distance_scale = args.distance_scale,
                    pos_pe = args.pos_pe, view_pe = args.view_pe, fea_pe = args.fea_pe, featureC = args.featureC, step_ratio = args.step_ratio, fea2denseAct = args.fea2denseAct)

    grad_vars = tensorf.get_optparam_groups()

    


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    #TODO: Do we need arg parser for lego training?
    args = config_parser()
    print(args)

    #TODO: mesh / render comes later
    reconstruction(args)

from re import M
import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--expname', type=str, 
                        help='experiment time')
    parser.add_argument('--basedir', type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument('--add_timestamp', type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument('--datadir', type=str, default='./data/liff/fern',
                        help='input data directory')
    parser.add_argument('--progress_refresh_rate', type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP'])
    
    # loader options
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_iters', type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'liff', 'nsvf', 'dtu', 'tankstemple', 'own_data'])

    # training options
    # learning rate
    parser.add_argument('--lr_init', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--lr_basis', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_iters', type=int, default=-1,
                        help='number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument('--lr_decay_target_ratio', type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument('--lr_upsample_reset', type=int, default=1,
                        help='reset lr to inital after upsampling')

    # model
    # volume options
    parser.add_argument('--n_lamb_sigma', type=int, action='append')
    parser.add_argument('--n_lamb_sh', type=int, action='append')
    parser.add_argument('--data_dim_color', type=int, default=27)
    parser.add_argument('--alpha_mask_thre', type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument('--distance_scale', type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument('--density_shift', type=float, default=-10, 
                        help='shift density in softplus; making density=0 when feature==0')


    # network decoder
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    
    parser.add_argument('--shadingMode', type=str, default='MLP_PE',
                        help='which shading mode to use')
    parser.add_argument('--pos_pe', type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument('--fea_pe', type=int, default=6,
                        help='number of pe for features')
    parser.add_argument('--view_pe', type=int, default=6,
                        help='number of pe for view')
    parser.add_argument('--featureC', type=int, default=128,
                        help='hidden feature channel in MLP')
    


    #rendering options
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio', type=float, default=0.5)
    parser.add_argument('--fea2denseAct', type=str, default='softplus')

    ## blender flags
    parser.add_argument('--upsamp_list', type=int, action='append')
    parser.add_argument('--update_AlphaMask_list', type=int, action='append')
    
    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)

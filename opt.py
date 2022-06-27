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

    #TODO: Continue training options
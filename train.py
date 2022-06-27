"""The current goal of this file is to re-implement (copy) the part of the code in original 
    tensoRF implemention that corresponding to train the model."""

from opt import config_parser #TODO: implement opt.py
from dataLoader import dataset_dict


def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, 
                            is_stack=False)
    test_dataset = datatset(args.datadir, split='test', downsample=args.downsample_train,
                            is_stack=True)
    


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    #TODO: Do we need arg parser for lego training?
    args = config_parser()
    print(args)

    #TODO: mesh / render comes later
    reconstruction(args)

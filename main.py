import argparse, torch
import utils
from utils import colors
import glo
from glo import interpolation, test, train


## -----------------------Setting GLO------------------------- ##
def parse_args():
    desc = "Main of Generative Latent Optimization"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s',        type=str, help='Which stage?', required=True)
    parser.add_argument('-date',     type=str, help='Date of this experiment', required=True)
    parser.add_argument('-test_data',type=str, help='Data in test stage')
    parser.add_argument('-dataset',  type=str, default='solar', help='The name of dataset', required=True)
    parser.add_argument('-p',        type=str, default='glo', help='Prefix of saved image')
    parser.add_argument('-dim',      type=int, default=100, help='Dimension of latent code')
    parser.add_argument('-e',        type=int, default=25, help='Nums of training epochs', required=True)
    parser.add_argument('-gpu',      type=bool, default=True, help='Use gpu?')
    parser.add_argument('-b',        type=int, default=128, help='Batch size')
    parser.add_argument('-lrg',      type=float, default=1., help='Learning rate of generator')
    parser.add_argument('-lrz',      type=float, default=10., help='Learning rate for representation space')
    parser.add_argument('-i',        type=str, default='pca', choices=['pca','random'], help='Init strategy for representation vectors')
    parser.add_argument('-l',        type=str, default='lap_l1', choices=['lap_l1','l2'], help='Loss type')
    parser.add_argument('-gpu_num',  type=int,  default=0,      help='which gpu?')
    return parser.parse_args()

## --------------------Input the Setting---------------------- ##
if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()

    # read the input
    date                = args.date
    dataset             = args.dataset
    image_output_prefix = args.p
    code_dim            = args.dim
    epochs              = args.e
    use_cuda            = args.gpu
    batch_size          = args.b
    lr_g                = args.lrg
    lr_z                = args.lrz
    init                = args.i
    loss                = args.l
    torch.cuda.set_device(args.gpu_num)
    # start training or testing
    if args.s == 'test':
        test_data           = args.test_date
        if args.test_data is None:
            raise Exception(colors.FAIL+"Must provide a data for test stage!!"+colors.ENDL)
        test(date,test_data,dataset, image_output_prefix, code_dim, epochs, use_cuda, batch_size, lr_g, lr_z, init, loss)
    elif args.s == 'ip':
        interpolation(date, dataset, image_output_prefix, code_dim, epochs, use_cuda, batch_size)
    elif args.s == 'train':
        train(        date, dataset, image_output_prefix, code_dim, epochs, use_cuda, batch_size, lr_g, lr_z, init, loss)
    else:
        raise Exception(colors.FAIL+"No such stage!!"+colors.ENDL)

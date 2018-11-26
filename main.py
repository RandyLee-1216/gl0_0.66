import argparse
import glo
from glo import test, train
import utils
from utils import colors

#-------------------------------------------------------------------------------
def parse_args():
    desc = "Main of Global Latent Optimization"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-dataset',  type=str,  required=True)
    parser.add_argument('-test_data',type=str)
    parser.add_argument('-date',     type=str,  required=True)
    parser.add_argument('-s',        type=str,  required=True)
    parser.add_argument('-p',        type=str,  default='glo')
    parser.add_argument('-dim',      type=int,  default=100)
    parser.add_argument('-e',        type=int,  required=True)
    parser.add_argument('-gpu',      type=bool, default=True)
    parser.add_argument('-b',        type=int,  default=128)
    parser.add_argument('-lrg',      type=float,default=.1)
    parser.add_argument('-lrz',      type=float,default=.1)
    parser.add_argument('-i',        type=str,  default='pca')
    parser.add_argument('-l',        type=str,  default='lap_l1', choices=['lap_l1','l2'])
    return parser.parse_args()

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()

    # start training or testing
    if args.s == 'test':
        if args.test_data is None:
            raise Exception(colors.FAIL+"Must provide a data for test stage!!"+colors.ENDL)
        test(date=args.date, test_data=args.test_data, dataset=args.dataset, image_output_prefix=args.p, code_dim=args.dim, epochs=args.e,
            use_cuda=args.gpu, batch_size=args.b, lr_g=args.lrg, lr_z=args.lrz, init=args.i, loss=args.l)
    elif args.s == 'train':
        train(date=args.date, dataset=args.dataset, image_output_prefix=args.p, code_dim=args.dim, epochs=args.e,
            use_cuda=args.gpu, batch_size=args.b, lr_g=args.lrg, lr_z=args.lrz, init=args.i, loss=args.l)
    else:
        raise Exception(colors.FAIL+"No such stage!!"+colors.ENDL)

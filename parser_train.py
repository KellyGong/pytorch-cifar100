import argparse
from train import train
import nni


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg16', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-early_stop', type=int, default=5, help='epoch of early stop')
    args = parser.parse_args()

    param_net = nni.get_next_parameter()['net']
    args.net = param_net
    print(f"***** Model: {args.net} *****")

    train(args)

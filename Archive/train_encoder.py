import random
import argparse
from distutils.util import strtobool

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import Encoder
from methods import initialize_weights
from data import case_118

"""settings"""
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed.')
parser.add_argument('--epoch', type=int, default=10000,  # epochs
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=16,  # embedded feature size
                    help='Number of hidden units.')
parser.add_argument('--n_head', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                    help='Initial learning rate.')
parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                    help="Toggle learning rate annealing for policy and value networks")
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
args = parser.parse_args()

"""seed"""
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

"""data"""
ind_load, load, ind_generator, generator, adjacent, _ = case_118()  # all in tensor format

"""model and optimizer"""
load = load.to(torch.float32)
encoder = Encoder(d_node=generator.shape[1],
                  d_hid=args.hidden,
                  n_head=args.n_head,
                  dropout=args.dropout,
                  alpha=args.alpha,)
encoder.apply(initialize_weights)
optimizer = Adam(params=encoder.parameters(),
                 lr=args.learning_rate,
                 eps=1e-5)


def train():
    encoder.train()
    train_loss = []
    for epoch in range(1, args.epoch+1):
        if args.anneal_lr:  # annealing the rate if instructed to do so
            frac = 1.0 - (epoch - 1.0) / args.epoch
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        feature = encoder(ind_load, load, ind_generator, generator, adjacent)

        loss = F.cross_entropy(torch.mm(feature, feature.t()), adjacent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.data)
    return train_loss


def inference():
    encoder.eval()
    feature = encoder(ind_load, load, ind_generator, generator, adjacent)
    return feature


if __name__ == '__main__':
    train_loss = train()

    x = [i for i in range(1, args.epoch+1)]
    plt.plot(x, train_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    embed_feature = inference()
    # save embed_feature
    torch.save(embed_feature, 'embed_feature.txt')

import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(".."))

from utils.utils import set_seed
from data.dataset_cifar import build_loader
from loss.loss_function import SDLoss
from loss.thr_loss import ThresholdScore
from loss.aps_loss import AdaptiveScoreFunction
import torch.optim as optim
from model.model_cifar100 import build_model
from model.logitmodel import build_logit_model
from utils.learning_utils import train_continuous



def main():
    parser = argparse.ArgumentParser(description='ConformalAdapter')
    parser.add_argument('--seed', type=int, default = 42, help='seed')
    parser.add_argument('--model', type=str, default='densenet121')
    parser.add_argument('--data_dir', '-s', type=str, default='../datasets', help='dataset name')
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, 
                        help="weight decay, can be tuned in (0,1e-4,1e-5,1e-6) for desired coverage rate")
    parser.add_argument('--ts', type=str, default='thr', help="training score”")
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='Device to run')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model weights')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    set_seed(args.seed)
    model_name = args.model
    device = args.device
    data_dir = args.data_dir
    lr = args.learning_rate
    weight_decay = args.weight_decay
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    save_dir = os.path.join(base_dir, args.save_dir)
    dataset = 'cifar100'
    class_num = 100

    model = build_model(model_name, use_adapter= False)
    model = model.to(device)
    trainloader = build_loader(data_dir, model, device= device, train= True)

    model = build_logit_model(class_num, use_adapter=True)
    model = model.to(device)
   
    save_iters = [20, 40, 60, 80, 100, 120]
    max_iter = save_iters[-1]+1

    if args.ts == 'thr':
        T = 1e-4
        scorefunction = ThresholdScore(T)
    else:
        scorefunction = AdaptiveScoreFunction()
    criterion = SDLoss(scorefunction)
    optimizer = optim.Adam(model.base_model.parameters(), lr=lr, weight_decay=weight_decay)
    train_continuous(model, trainloader, criterion, optimizer, model_name, dataset, max_iter, save_iters, save_dir, device)
            




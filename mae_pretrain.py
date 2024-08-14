import os
import argparse
import math
import torch
import torchvision
import wandb
import cv2
wandb.require("core")

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from data_utils import get_loader

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--dataset', type=str, default='polyvore')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    wandb.login(key='3b3fd7ec86b8f3f0f32f2d7a78456686d8755d99')
    wandb.init(project=f'mae_{args.dataset}', name=f'{args.dataset}')

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dir = f'{args.dataset}_train'
    os.makedirs(train_dir, exist_ok=True)

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    elif args.dataset == 'polyvore':
        train_dataset, val_dataset = get_loader(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(image_size=args.img_size, patch_size=16, mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_dataset)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        wandb.log({'train_loss': avg_loss})
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        cnt = 0
        with torch.no_grad():
            val_data_loader = val_dataset
            for val_img, label in val_data_loader:
                val_img = val_img.to(device)
                predicted_val_img, mask = model(val_img)
                predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
                img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
                img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                img = (img + 1) / 2
                img = img.cpu().detach().numpy()
                cv2.imwrite(os.path.join(train_dir, f'val_{cnt}.jpg'), img)
                cnt += 1

        ''' save model '''
        torch.save(model, os.path.join(train_dir, args.model_path))
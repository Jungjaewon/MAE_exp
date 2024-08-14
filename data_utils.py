import logging
import os
import torch
import os.path as osp
import random, pickle

from PIL import Image
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset


logger = logging.getLogger(__name__)

class Classification_2d(Dataset):

    def __init__(self, transform, mode):
        self.image_dir = osp.join('outfitdata_set3_4598', mode)
        self.transform = transform
        self.mode = mode

        self.outfit_list = glob(osp.join(self.image_dir, '*'))
        self.outfit_data = list()
        random.shuffle(self.outfit_list)

        with open('outfitdata_set3_tagged.plk', 'rb') as fp:
            self.tagged_dict = pickle.load(fp)[mode]

        for outfit_id_path in self.outfit_list:
            outfit_id = outfit_id_path.split(os.sep)[-1]
            for i in range(1, 6):
                data_path = osp.join(outfit_id_path, f'{i}.jpg')
                cat_idx = self.tagged_dict[f'{outfit_id}_{str(i)}']['cate_idx']
                self.outfit_data.append([data_path, cat_idx])

    def __getitem__(self, index):
        data = self.outfit_data[index]
        img_path = data[0]
        cate_idx = data[1]
        target_image = Image.open(osp.join(img_path))
        target_image = target_image.convert('RGB')
        return self.transform(target_image), cate_idx#torch.LongTensor([cate_idx])

    def __len__(self):
        return len(self.outfit_data)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == 'polyvore':
        trainset = Classification_2d(transform_train, 'train')
        testset = Classification_2d(transform_test, 'test')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

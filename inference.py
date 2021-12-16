#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')


from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.utils import *
from timm.loss import *

import torchvision

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    print(img.size())
    print(img.min(), img.max())
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def imshow_cpu(img):
    print(img.shape)
    print(img.min(), img.max())
    img = img/255     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    
def imshow_pil(img):
    img = np.array(img)
    print(img.shape)
    print(img.min(), img.max())
    img = img     # unnormalize
    plt.imshow(img)
    plt.show()


# from torchvision.io import read_image
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from PIL import Image
# import numpy as np
# from torchvision import datasets
# from torchvision import transforms
# from torch.utils.data.sampler import SubsetRandomSampler
# from pathlib import Path

# class CustomBinary(Dataset):
#     def __init__(self, data_dir, transform=None):
#         data_dir = Path(data_dir)
#         self.true = list(data_dir.joinpath('true').rglob('*'))
#         self.false = list(data_dir.joinpath('false').rglob('*'))
# #         self.height = 48
# #         self.width = 48
#         self.transform = transform

#     def __getitem__(self, index):
#         # This method should return only 1 sample and label 
#         # (according to "index"), not the whole dataset
#         # So probably something like this for you:
#         if index < len(self.true):
#             image = np.array(Image.open(self.true[index]))
# #             image = read_image(str(self.true[index]))
#             label = True
#         else:
#             image = np.array(Image.open(self.false[index-len(self.true)]))
# #             image = read_image(str(self.false[index-len(self.true)]))
#             label = False
            
#         return image, label

#     def __len__(self):
#         return len(self.true)+len(self.false)
    
# def load_dataset(data_dir, batch_size=50, validation_split=.2, shuffle_dataset=True, random_seed=42):
#     dataset = CustomBinary(data_dir)

# #     batch_size = 50
# #     validation_split = .2
# #     shuffle_dataset = True
# #     random_seed= 42

#     # Creating data indices for training and validation splits:
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(validation_split * dataset_size))
#     if shuffle_dataset :
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)
#     train_indices, val_indices = indices[split:], indices[:split]

#     # Creating PT data samplers and loaders:
#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)

# #     train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
# #                                                sampler=train_sampler)
# #     validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
# #                                                     sampler=valid_sampler)

#     return train_sampler, valid_sampler


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    print('Test time pool',test_time_pool)
    
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()
        
        
    # create the train and eval datasets
#     dataset_train = create_dataset(
#         args.dataset,
#         root=args.data, split=args.train_split, is_training=True,
#         batch_size=args.batch_size, repeats=args.epoch_repeats)
    
    dataset_eval = create_dataset(
        args.dataset, root=args.data, split='test', is_training=False, batch_size=args.batch_size, transform=None)
    
#     imshow_pil(dataset_eval[0][0])
#     import pdb
#     pdb.set_trace()
        
#     train_sampler, valid_sampler = load_dataset(args.data, batch_size=args.batch_size, validation_split=.01, shuffle_dataset=True, random_seed=42)

    print(config['crop_pct'])
    crop_pct = config['crop_pct']
    
    fos = config['input_size']
    print(fos)
    print(fos[-2:])
    
    import math
    
    scale_size = int(math.floor(fos[-2:][0] / crop_pct))
    
    print(scale_size)

    loader = create_loader(
#         ImageDataset(args.data),
        dataset_eval,
        no_aug=True,
#         sampler = valid_sampler,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=None)
#         crop_pct=1.0 if test_time_pool else config['crop_pct'])
    
    print('Dataset len {} dataloader len {}'.format(len(dataset_eval), len(loader)))

    model.eval()
    
#     count_True = 0
#     count_False = 0
#     for img, target in dataset_eval:
#         if target:
#             count_True = count_True+1
#         else:
#             count_False = count_False+1
            

            
#     count_True_l = 0
#     count_False_l = 0
#     for batch_idx, (input, target) in enumerate(loader):
#         count_True_l = count_True_l+torch.sum(target)
#         count_False_l = count_False_l+(len(target)-torch.sum(target))
    
    classes = ['false', 'true']
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    filenames = loader.dataset.filenames(basename=False)
    

#     imshow_cpu(loader.dataset[0][0])


#     import pdb
#     pdb.set_trace()
    
    correct = 0
    total = 0

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    end = time.time()
    topk_ids = []
    all_target = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
#             pdb.set_trace()
            input = input.cuda()
            target = target.cuda()
            labels = model(input)
            
            
#             imshow(torchvision.utils.make_grid(input))
#             print(filenames[batch_idx])
            
#             print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#             import pdb
#             pdb.set_trace()
            
            
            _, predictions = torch.max(labels, 1)
            
            total += labels.size(0)
            correct += (predictions == target).sum().item()
            
            # collect the correct predictions for each class
            for label, prediction in zip(target, predictions):
                if target == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            
#             import pdb
#             pdb.set_trace()
            
            acc1, acc5 = accuracy(labels, target, topk=(1, 5))
            
            top1_m.update(acc1.item(), labels.size(0))
            
#             pdb.set_trace()
            topk = labels.topk(1)[1]
            topk_ids.append(topk.cpu().numpy())
        
            all_target.extend(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})  Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})'.format(
                    batch_idx, len(loader), batch_time=batch_time, top1=top1_m))

    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        class_accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       class_accuracy))
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    
    
    topk_ids = np.concatenate(topk_ids, axis=0)
    
#     pdb.set_trace()

    with open(os.path.join(args.output_dir, './topk_ids.csv'), 'w') as out_file:
        filenames = loader.dataset.filenames(basename=False)
        for filename, label, tt in zip(filenames, topk_ids, all_target):
            out_file.write('{0},{1},{2}\n'.format(
                filename, ','.join([ str(v) for v in label]), str(tt)))


if __name__ == '__main__':
    main()

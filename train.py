import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from src.utils.losses import BCEDiceLoss
from src.utils.meter import AverageMeter
from src.utils.metrics import iou_score
from src.utils.augment import medical_augment
from src.utils import ramps
from src.dataloader.dataset import (SemiDataSets, TwoStreamBatchSampler)
from src.network.DEMS import DEMS
from torch.optim.lr_scheduler import CosineAnnealingLR


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--semi_percent', type=float, default=0.8, help='percentage of labeled images')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="train_1.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="val_1.txt", help='dir')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number')
parser.add_argument('--total_batch_size', type=int, default=8, help='batch size')
parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled batch size')
parser.add_argument('--consistency', type=float, default=7, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency rampup')
parser.add_argument('--kernel_size', type=int, default=7, help='RREC kernel size')
parser.add_argument('--length', type=tuple, default=(3, 3, 3), help='length of RREC')
args = parser.parse_args()

seed_torch(args.seed)


def getDataloader(args):
    train_transform = Compose([
        medical_augment(level=5),
        transforms.Normalize(),
    ])
    val_transform = Compose([
        transforms.Normalize(),
    ])
    labeled_slice = args.semi_percent
    db_train = SemiDataSets(base_dir=args.base_dir, split="train", transform=train_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = SemiDataSets(base_dir=args.base_dir, split="val", transform=val_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = len(db_train)
    labeled_idxs = list(range(0, int(labeled_slice * total_slices)))
    unlabeled_idxs = list(range(int(labeled_slice * total_slices), total_slices))
    print("Labeled: {} ({}%), unlabeled: {}".format(len(labeled_idxs), labeled_slice * 100, len(unlabeled_idxs)))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.total_batch_size, args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    return trainloader, valloader


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args):
    base_lr = args.base_lr
    max_iterations = int(args.max_iterations * args.semi_percent)
    trainloader, valloader = getDataloader(args)
    model = DEMS(length=args.length, k=args.kernel_size).cuda()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = BCEDiceLoss().cuda()

    best_iou, iter_num = 0, 0
    max_epoch = max_iterations // len(trainloader) + 1
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, verbose=False)
    for epoch_num in range(max_epoch):
        avg_meters = {'train_loss': AverageMeter(),
                      'fus_loss': AverageMeter(),
                      'sen_loss': AverageMeter(),
                      'uns_loss': AverageMeter(),
                      'train_iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_dsc': AverageMeter(),
                      'val_sen': AverageMeter(),
                      'val_pre': AverageMeter(),
                      'val_fos': AverageMeter(),
                      'val_spe': AverageMeter(),
                      'val_acc': AverageMeter()
                      }

        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs_main, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)
            outputs_main_soft = torch.sigmoid(outputs_main)
            outputs_aux1_soft = torch.sigmoid(outputs_aux1)
            outputs_aux2_soft = torch.sigmoid(outputs_aux2)
            outputs_aux3_soft = torch.sigmoid(outputs_aux3)

            loss_fus = criterion(outputs_main[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_fus_aux1 = criterion(outputs_aux1[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_fus_aux2 = criterion(outputs_aux2[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_fus_aux3 = criterion(outputs_aux3[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            fus_loss = (loss_fus + loss_fus_aux1 + loss_fus_aux2 + loss_fus_aux3) / 4

            loss_uns_aux1 = torch.mean((outputs_main_soft[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2)
            loss_uns_aux2 = torch.mean((outputs_main_soft[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2)
            loss_uns_aux3 = torch.mean((outputs_main_soft[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2)
            uns_loss = (loss_uns_aux1 + loss_uns_aux2 + loss_uns_aux3) / 3

            output_main_bri = (outputs_main_soft[:args.labeled_bs] > 0.5).float()
            output_aux1_bri = (outputs_aux1_soft[:args.labeled_bs] > 0.5).float()
            output_aux2_bri = (outputs_aux2_soft[:args.labeled_bs] > 0.5).float()
            output_aux3_bri = (outputs_aux3_soft[:args.labeled_bs] > 0.5).float()

            diff_area_ma1 = torch.sum(((output_main_bri == 1) ^ (output_aux1_bri == 1)).float())
            diff_area_ma2 = torch.sum(((output_main_bri == 1) ^ (output_aux2_bri == 1)).float())
            diff_area_ma3 = torch.sum(((output_main_bri == 1) ^ (output_aux3_bri == 1)).float())

            total_pixels = torch.sum(label_batch[:args.labeled_bs] == 1) + torch.sum(label_batch[:args.labeled_bs] == 0)
            sen_loss_ma1 = diff_area_ma1 / (total_pixels + 1e-8)
            sen_loss_ma2 = diff_area_ma2 / (total_pixels + 1e-8)
            sen_loss_ma3 = diff_area_ma3 / (total_pixels + 1e-8)
            sen_loss = (sen_loss_ma1 + sen_loss_ma2 + sen_loss_ma3) / 3

            weight = get_current_consistency_weight(iter_num // 150)
            loss = fus_loss + weight * sen_loss + weight * uns_loss
            iou, _, _, _, _, _, _ = iou_score(outputs_main[:args.labeled_bs], label_batch[:args.labeled_bs])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['train_loss'].update(loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['fus_loss'].update(fus_loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['sen_loss'].update(sen_loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['uns_loss'].update(uns_loss.item(), volume_batch[args.labeled_bs:].size(0))
            avg_meters['train_iou'].update(iou, volume_batch[:args.labeled_bs].size(0))

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['val_dsc'].update(dice, input.size(0))
                avg_meters['val_sen'].update(SE, input.size(0))
                avg_meters['val_pre'].update(PC, input.size(0))
                avg_meters['val_fos'].update(F1, input.size(0))
                avg_meters['val_spe'].update(SP, input.size(0))
                avg_meters['val_acc'].update(ACC, input.size(0))

        save_str = f"DEMS_{os.path.basename(args.base_dir)}_{args.semi_percent}_{args.seed}"

        print(
            'Epoch [%3d/%d] Train: L %.4f, Lf %.4f, Ls %.4f, Lu %.4f, IoU %.4f; Validation: L %.4f, IoU %.4f, '
            'DSC %.4f, SEN %.4f, PRE %.4f, FOS %.4f, SPE %.4f, ACC %.4f'
            % (epoch_num+1, max_epoch, avg_meters['train_loss'].avg, avg_meters['fus_loss'].avg,
               avg_meters['sen_loss'].avg, avg_meters['uns_loss'].avg, avg_meters['train_iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dsc'].avg,
               avg_meters['val_sen'].avg, avg_meters['val_pre'].avg, avg_meters['val_fos'].avg, avg_meters['val_spe'].avg,
               avg_meters['val_acc'].avg), file=open(f"./checkpoint/{save_str}_log.txt", "a"))

        if avg_meters['val_iou'].avg > best_iou:
            torch.save(model.state_dict(), f'checkpoint/{save_str}_model.pth')
            torch.save(model, f'checkpoint/{save_str}_model.pkl')
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model", file=open(f"./checkpoint/{save_str}_log.txt", "a"))

    return "Training Finished!"


if __name__ == "__main__":
    train(args)

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data
from tensorboardX import SummaryWriter
import argparse
import datetime
import os
import glob

from utils import timer
from modules.build_yolact import Yolact
from modules.multi_loss import Multi_Loss
from data.config import get_config
from data.coco import COCODetection
from eval import evaluate
from data.coco import detection_collate

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--cfg', default='res101_coco', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
parser.add_argument('--test_bs', type=int, default=1, help='-1 to disable val')
parser.add_argument('--img_size', default=550, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=4000, type=int,
                    help='The validation interval during training, pass -1 to disable.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def data_to_device(datum):
    images, targets, masks, num_crowds = datum

    if cuda:
        images = images.cuda().detach()
        targets = [ann.cuda().detach() for ann in targets]
        masks = [mask.cuda().detach() for mask in masks]
    else:
        images = images.detach()
        targets = [ann.detach() for ann in targets]
        masks = [mask.detach() for mask in masks]

    return images, targets, masks, num_crowds


def save_best(net, mask_map):
    weight = glob.glob('weights/best*')
    best_mask_map = float(weight[0].split('/')[-1].split('_')[1]) if weight else 0.

    if mask_map >= best_mask_map:
        if weight:
            os.remove(weight[0])  # remove the last best model

        print(f'\nSaving the best model as \'best_{mask_map}_{cfg_name}_{step}.pth\'.\n')
        torch.save(net.state_dict(), f'weights/best_{mask_map}_{cfg_name}_{step}.pth')


def save_latest(net):
    weight = glob.glob('weights/latest*')
    if weight:
        os.remove(weight[0])

    torch.save(net.state_dict(), f'weights/latest_{cfg_name}_{step}.pth')


class NetWithLoss(nn.Module):
    def __init__(self, net, loss):
        super().__init__()
        self.net = net
        self.loss = loss

    def forward(self, images, box_classes, masks_gt, num_crowds):
        predictions = self.net(images)
        return self.loss(predictions, box_classes, masks_gt, num_crowds)


args = parser.parse_args()
cfg = get_config(args)

cuda = torch.cuda.is_available()
if cuda:
    main_gpu = dist.get_rank() == 0
    num_gpu = dist.get_world_size()

net = Yolact(cfg)
net.train()
optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
criterion = Multi_Loss(cfg)

if args.resume == 'latest':
    weight = glob.glob('weights/latest*')[0]
    net.load_weights(weight, cuda)
    start_step = int(weight.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{weight}\'.\n')
elif args.resume and 'yolact' in args.resume:
    net.load_weights(cfg.weight, cuda)
    start_step = int(cfg.weight.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{args.resume}\'.\n')
else:
    net.init_weights(cfg.weight)
    print(f'\nTraining from begining, weights initialized with {cfg.weight}.\n')
    start_step = 0

dataset = COCODetection(cfg, val=False)
train_sampler = None
if cuda:
    cudnn.benchmark = True
    net_with_loss = NetWithLoss(net, criterion)
    net = DDP(net_with_loss.cuda(), [args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
    train_sampler = DistributedSampler(dataset, shuffle=True)

# shuffle must be False if sampler is specified
data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=cfg.bs_per_gpu, shuffle=(train_sampler is None),
                              collate_fn=detection_collate, pin_memory=True, sampler=train_sampler)

step_index = 0
epoch_seed = 0
map_tables = []
training = True
step = start_step
cfg_name = cfg.__class__.__name__
timer.reset()
writer = SummaryWriter('tensorboard_log')

try:  # Use try-except to use ctrl+c to stop and save early.
    while training:
        if train_sampler:
            epoch_seed += 1
            train_sampler.set_epoch(epoch_seed)

        for i, datum in enumerate(data_loader):
            if ((not cuda) or main_gpu) and i == start_step + 1:
                timer.start()

            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # Warm up learning rate.
                set_lr(optimizer, (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init)

            # Adjust the learning rate according to the current step.
            while step_index < len(cfg.lr_steps) and step >= cfg.lr_steps[step_index]:
                step_index += 1
                set_lr(optimizer, cfg.lr * (0.1 ** step_index))

            images, box_classes, masks_gt, num_crowds = data_to_device(datum)

            with timer.counter('for+loss'):
                loss_b, loss_m, loss_c, loss_s = net(images, box_classes, masks_gt, num_crowds)

                if cuda:
                    # dist.reduce() should not be called in `if main_gpu:`, or processes
                    # would be waiting forever.
                    all_loss = torch.stack([loss_b, loss_m, loss_c, loss_s], dim=0)
                    dist.reduce(all_loss, dst=0)

                    # Get the mean loss across all GPUS, this is for printing and has nothing to do with
                    # the model optimization.
                    if main_gpu:
                        l_b = all_loss[0].item() / num_gpu  # seems when printing, need to call .item(), not sure
                        l_m = all_loss[1].item() / num_gpu
                        l_c = all_loss[2].item() / num_gpu
                        l_s = all_loss[3].item() / num_gpu

            with timer.counter('backward'):
                loss_total = loss_b + loss_m + loss_c + loss_s
                optimizer.zero_grad()
                loss_total.backward()

            with timer.counter('update'):
                if torch.isfinite(loss_total):
                    optimizer.step()
                else:
                    print(loss_total)

            time_this = time.time()
            if i > start_step:
                batch_time = time_this - time_last
                timer.add_batch_time(batch_time)
            time_last = time_this

            if step % 20 == 0 and step != start_step:
                if (not cuda) or main_gpu:
                    cur_lr = optimizer.param_groups[0]['lr']
                    time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
                    t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                    seconds = (cfg.max_iter - step) * t_t
                    eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                    writer.add_scalar(f'task/box', l_b, global_step=step)
                    writer.add_scalar(f'task/mask', l_m, global_step=step)
                    writer.add_scalar(f'task/class', l_c, global_step=step)
                    writer.add_scalar(f'task/semantic', l_s, global_step=step)
                    writer.add_scalar('total', loss_total, global_step=step)

                    print(f'step: {i} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | '
                          f'l_mask: {l_m:.3f} | l_semantic: {l_s:.3f} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | '
                          f't_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

            if args.val_interval > 0 and step % args.val_interval == 0 and step != start_step:
                if (not cuda) or main_gpu:
                    net.eval()
                    table, box_row, mask_row = evaluate(net.module.net, cfg)
                    net.train()

                    writer.add_scalar('box_map', box_row[1], global_step=step)
                    writer.add_scalar('mask_map', mask_row[1], global_step=step)

                    map_tables.append(table)
                    save_latest(net.module.net if cuda else net)
                    save_best(net.module.net if cuda else net, mask_row[1])

                    timer.reset()  # training time and val time share the same Obj, so reset it to avoid confusion

            if ((not cuda) or main_gpu) and i != 1 and i % cfg.val_interval == 1:
                timer.start()  # the first iter after validation should not be included

            step += 1

            if step > cfg.max_iter:
                training = False

                if (not cuda) or main_gpu:
                    print(f'Training completed, saving the final model as \'latest_{cfg_name}_{step}.pth\'.\n')
                    save_latest(net.module.net if cuda else net)

                break

except KeyboardInterrupt:
    if (not cuda) or main_gpu:
        print(f'\nStopped, saving the latest model as \'latest_{cfg_name}_{step}.pth\'.\n')
        save_latest(net.module.net if cuda else net)

        print('\nValidation results during training:\n')
        for table in map_tables:
            print(table, '\n')

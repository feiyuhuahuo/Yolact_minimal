from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage
from utils import timer
from modules.build_yolact import Yolact
import time
import torch
from modules.multi_loss import Multi_Loss
from data.config import set_cfg, cfg
from data.coco import COCODetection
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import datetime
from eval import evaluate
import os
from data.coco import detection_collate

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--config', default='yolact_base_config', help='The config object to use.')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--resume', default=None, type=str, help='The path of checkpoint file to resume training from.')
parser.add_argument('--val_interval', default=10000, type=int,
                    help='Val and save the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--max_keep', default=10, type=int, help='The maximum number of .pth files to keep.')


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def data_to_device(datum):
    images, targets, masks, num_crowds = datum

    if cuda:
        images = Variable(images.cuda(), requires_grad=False)
        targets = [Variable(ann.cuda(), requires_grad=False) for ann in targets]
        masks = [Variable(mask.cuda(), requires_grad=False) for mask in masks]
    else:
        images = Variable(images, requires_grad=False)
        targets = [Variable(ann, requires_grad=False) for ann in targets]
        masks = [Variable(mask, requires_grad=False) for mask in masks]

    return images, targets, masks, num_crowds


def compute_val_map(yolact_net):
    with torch.no_grad():
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    augmentation=BaseTransform())
        yolact_net.eval()
        print("\nComputing validation mAP...", flush=True)
        table = evaluate(yolact_net, val_dataset, during_training=True)
        yolact_net.train()
        return table


def print_result(map_tables):
    print('\nValidation results during training:\n')
    for info, table in map_tables:
        print(info)
        print(table, '\n')


def save_weights(net, epoch, iteration):
    net.module.save_weights(f'weights/{cfg.name}_{epoch}_{iteration}.pth')

    path_list = os.listdir('weights')
    path_list = [aa for aa in path_list if 'yolact_base' in aa]
    path_list.remove('yolact_base_54_800000.pth')
    iter_num = [int(aa.split('.')[0].split('_')[-1]) for aa in path_list]
    iter_num.sort()

    if len(path_list) > args.max_keep:
        for aa in path_list:
            if str(iter_num[0]) in aa:
                os.remove('weights/' + aa)


args = parser.parse_args()
if args.config is not None:
    set_cfg(args.config)

print(vars(args), '\n')

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset = COCODetection(image_path=cfg.dataset.train_images,
                        info_file=cfg.dataset.train_info,
                        augmentation=SSDAugmentation())

net = Yolact()
net.train()

# Don't use the timer during training, there's a race condition with multiple GPUs.
timer.disable_all()

if args.resume is not None:
    net.load_weights(args.resume)
    resume_epoch = int(args.resume.split('.')[0].split('_')[2])
    resume_iter = int(args.resume.split('.')[0].split('_')[3])
    print(f'\nResume training at epoch: {resume_epoch}, iteration: {resume_iter}.')
else:
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)
    print('\nTraining from begining, weights initialized.')

optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.decay)
criterion = Multi_Loss(num_classes=cfg.num_classes,
                       pos_thre=cfg.positive_iou_threshold,  # 0.5
                       neg_thre=cfg.negative_iou_threshold,  # 0.4
                       negpos_ratio=3)

if cuda:
    cudnn.benchmark = True
    net = nn.DataParallel(net).cuda()
    criterion = nn.DataParallel(criterion).cuda()

epoch_size = len(dataset) // args.batch_size
step_index = 0

iteration = resume_iter if args.resume else 0
start_epoch = iteration // epoch_size
end_epoch = cfg.max_iter // epoch_size + 1
remain = epoch_size - (iteration % epoch_size)

data_loader = data.DataLoader(dataset, args.batch_size, num_workers=8, shuffle=True,
                              collate_fn=detection_collate, pin_memory=True)

batch_time = MovingAverage()
loss_types = ['B', 'C', 'M', 'S']
loss_avgs = {k: MovingAverage(100) for k in loss_types}
map_tables = []

print('Begin training!\n')
# Use try-except to use ctrl+c to stop and save early.
try:
    for epoch in range(start_epoch, end_epoch):
        for i, datum in enumerate(data_loader):
            if args.resume and epoch == start_epoch and i >= remain:
                break

            iteration += 1

            # Warm up learning rate
            if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                set_lr(optimizer,
                       (cfg.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

            # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
            while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                step_index += 1
                set_lr(optimizer, cfg.lr * (0.1 ** step_index))

            images, box_classes, masks_gt, num_crowds = data_to_device(datum)

            torch.cuda.synchronize()
            forward_start = time.time()

            predictions = net(images)

            torch.cuda.synchronize()
            forward_end = time.time()

            losses = criterion(predictions, box_classes, masks_gt, num_crowds)
            losses = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])

            optimizer.zero_grad()
            loss.backward()  # Do this to free up vram even if loss is not finite
            if torch.isfinite(loss).item():
                optimizer.step()

            # Add the loss to the moving average for bookkeeping
            for k in losses:
                loss_avgs[k].add(losses[k].item())

            grad_end = time.time()
            if not (i == 0 and epoch == start_epoch):
                iter_time = grad_end - temp
                batch_time.add(iter_time)
            temp = grad_end

            if iteration % 10 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                eta_str = str(datetime.timedelta(seconds=(cfg.max_iter - iteration) * batch_time.get_avg())).split('.')[
                    0]
                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                forward_time = forward_end - forward_start
                data_time = iter_time - (grad_end - forward_start)
                print(('[%3d] %7d |' + (' %s: %.3f |' * len(
                    losses)) + ' T: %.3f | lr: %.5f | t_data: %.3f | t_forward: %.3f | t_total: %.3f | ETA: %s')
                      % tuple(
                    [epoch, iteration] + loss_labels + [total, cur_lr, data_time, forward_time, iter_time, eta_str]),
                      flush=True)

            if args.val_interval > 0 and iteration % args.val_interval == 0:
                print(f'Saving network at epoch: {epoch}, iteration: {iteration}.\n')
                save_weights(net, epoch, iteration)

                info = (('iteration: %7d |' + (' %s: %.3f |' * len(losses)) + ' T: %.3f | lr: %.5f')
                        % tuple([iteration] + loss_labels + [total, cur_lr]))
                table = compute_val_map(net.module)
                map_tables.append((info, table))

except KeyboardInterrupt:
    print(f'\nStopped, saving network at epoch: {epoch}, iteration: {iteration}.\n')
    save_weights(net, epoch, iteration)

    print_result(map_tables)
    exit()

print(f'Training completed, saving network at epoch: {epoch}, iteration: {iteration}.\n')
save_weights(net, epoch, iteration)

print_result(map_tables)

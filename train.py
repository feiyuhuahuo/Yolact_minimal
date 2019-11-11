import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import datetime
import os
import glob

from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage
from utils import timer
from modules.build_yolact import Yolact
from modules.multi_loss import Multi_Loss
from data.config import cfg, update_config
from data.coco import COCODetection
from eval import evaluate
from data.coco import detection_collate

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--config', default='yolact_base_config', help='The config object to use.')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--img_size', default=550, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=10000, type=int,
                    help='Evalute and save the model every [val_interval] iterations, pass -1 to disable.')


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


def compute_val_map(yolact_net):
    with torch.no_grad():
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    augmentation=BaseTransform())
        yolact_net.eval()
        print("\nComputing validation mAP...", flush=True)
        table, mask_row = evaluate(yolact_net, val_dataset, during_training=True)
        yolact_net.train()
        return table, mask_row[1]


def print_result(map_tables):
    print('\nValidation results during training:\n')
    for info, table in map_tables:
        print(info)
        print(table, '\n')


args = parser.parse_args()
update_config(args.config, args.batch_size, args.img_size)
print('\n' + '-' * 30 + 'Configs' + '-' * 30)
for k, v in vars(cfg).items():
    print(f'{k}: {v}')

# Don't use the timer during training, there's a race condition with multiple GPUs.
timer.disable_all()

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

net = Yolact()
net.train()

if args.resume == 'latest':
    weight = glob.glob('weights/latest*')[0]
    net.load_weights(weight)
    resume_iter = int(weight.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{weight}\'.\n')
elif args.resume and 'yolact' in args.resume:
    net.load_weights('weights/' + args.resume)
    resume_iter = int(args.resume.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{args.resume}\'.\n')
else:
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)
    print('\nTraining from begining, weights initialized.\n')

optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.decay)
criterion = Multi_Loss(num_classes=cfg.num_classes,
                       pos_thre=cfg.positive_iou_threshold,
                       neg_thre=cfg.negative_iou_threshold,
                       negpos_ratio=3)

if cuda:
    cudnn.benchmark = True
    net = nn.DataParallel(net).cuda()
    criterion = nn.DataParallel(criterion).cuda()

dataset = COCODetection(image_path=cfg.dataset.train_images,
                        info_file=cfg.dataset.train_info,
                        augmentation=SSDAugmentation())

epoch_size = len(dataset) // cfg.batch_size
step_index = 0

iter = resume_iter if args.resume else 0
start_epoch = iter // epoch_size
end_epoch = cfg.max_iter // epoch_size + 1
remain = epoch_size - (iter % epoch_size)

data_loader = data.DataLoader(dataset, cfg.batch_size, num_workers=8, shuffle=True,
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

            iter += 1

            # Warm up learning rate
            if cfg.warmup_until > 0 and iter <= cfg.warmup_until:
                set_lr(optimizer, (cfg.lr - cfg.warmup_init) * (iter / cfg.warmup_until) + cfg.warmup_init)

            # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
            while step_index < len(cfg.lr_steps) and iter >= cfg.lr_steps[step_index]:
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

            for k in losses:
                loss_avgs[k].add(losses[k].item())  # Add the loss to the moving average for bookkeeping

            grad_end = time.time()
            if i == 0 and epoch == start_epoch:
                temp = forward_start
            iter_time = grad_end - temp
            batch_time.add(iter_time)
            temp = grad_end

            if iter % 10 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                seconds = (cfg.max_iter - iter) * batch_time.get_avg()
                eta_str = str(datetime.timedelta(seconds=seconds)).split('.')[0]
                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                t_forward = forward_end - forward_start
                t_data = iter_time - (grad_end - forward_start)
                time_str = ' T: %.3f | lr: %.5f | t_data: %.3f | t_forward: %.3f | t_total: %.3f | ETA: %s'
                print(('[%3d] %7d |' + (' %s: %.3f |' * len(losses)) + time_str) % tuple(
                    [epoch, iter] + loss_labels + [total, cur_lr, t_data, t_forward, iter_time, eta_str]), flush=True)

            if args.val_interval > 0 and iter % args.val_interval == 0:
                info = (('iteration: %7d |' + (' %s: %.3f |' * len(losses)) + ' T: %.3f | lr: %.5f')
                        % tuple([iter] + loss_labels + [total, cur_lr]))
                table, mask_map = compute_val_map(net.module)

                weight = glob.glob('weights/best*')[0]
                best_mask_map = float(weight.split('/')[-1].split('_')[1]) if weight else 0.

                if mask_map >= best_mask_map:
                    print(f'Saving the current best model as \'best_{mask_map}_{cfg.name}_{epoch}_{iter}.pth\'.\n')
                    if weight:
                        os.remove(weight[0])
                    net.module.save_weights(f'weights/best_{mask_map}_{cfg.name}_{epoch}_{iter}.pth')

                map_tables.append((info, table))

except KeyboardInterrupt:
    print(f'\nStopped, saving the latest model as \'latest_{cfg.name}_{epoch}_{iter}.pth\'.\n')
    weight = glob.glob('weights/latest*')
    if weight:
        os.remove(weight[0])

    net.module.save_weights(f'weights/latest_{cfg.name}_{epoch}_{iter}.pth')
    print_result(map_tables)
    exit()

print(f'Training completed, saving the final model as \'{cfg.name}_{epoch}_{iter}.pth\'.\n')
net.module.save_weights(f'weights/{cfg.name}_{epoch}_{iter}.pth')

print_result(map_tables)

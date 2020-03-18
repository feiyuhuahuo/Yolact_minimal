import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from tensorboardX import SummaryWriter
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
parser.add_argument('--config', default='res101_coco_config', help='The config object to use.')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--img_size', default=550, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=10000, type=int,
                    help='Evaluate and save the model every [val_interval] iterations, pass -1 to disable.')


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
        table, box_row, mask_row = evaluate(yolact_net, val_dataset, during_training=True)
        yolact_net.train()
        return table, box_row[1], mask_row[1]


def print_result(map_tables):
    print('\nValidation results during training:\n')
    for info, table in map_tables:
        print(info)
        print(table, '\n')


def save_best(net):
    weight = glob.glob('weights/best*')
    best_mask_map = float(weight[0].split('/')[-1].split('_')[1]) if weight else 0.
    if mask_map >= best_mask_map:
        if weight:
            os.remove(weight[0])  # remove the last best model
        print(f'\nSaving the current best model as \'best_{mask_map}_{cfg.name}_{step}.pth\'.\n')
        torch.save(net.state_dict(), f'weights/best_{mask_map}_{cfg.name}_{step}.pth')


def save_latest(net):
    weight = glob.glob('weights/latest*')
    if weight:
        os.remove(weight[0])
    torch.save(net.state_dict(), f'weights/latest_{cfg.name}_{step}.pth')


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
    net.load_weights(weight, cuda)
    resume_step = int(weight.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{weight}\'.\n')
elif args.resume and 'yolact' in args.resume:
    net.load_weights('weights/' + args.resume, cuda)
    resume_step = int(args.resume.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{args.resume}\'.\n')
else:
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)
    print('\nTraining from begining, weights initialized.\n')

optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.decay)
criterion = Multi_Loss(num_classes=cfg.num_classes, pos_thre=cfg.pos_iou_thre, neg_thre=cfg.neg_iou_thre, np_ratio=3)

if cuda:
    cudnn.benchmark = True
    net = nn.DataParallel(net).cuda()
    criterion = nn.DataParallel(criterion).cuda()

dataset = COCODetection(image_path=cfg.dataset.train_images, info_file=cfg.dataset.train_info,
                        augmentation=SSDAugmentation())

data_loader = data.DataLoader(dataset, cfg.batch_size, num_workers=8, shuffle=True,
                              collate_fn=detection_collate, pin_memory=True)

step_index = 0
start_step = resume_step if args.resume else 0
batch_time = MovingAverage()
loss_types = ['B', 'C', 'M', 'S']
loss_avgs = {k: MovingAverage() for k in loss_types}
map_tables = []
training = True
step = start_step
writer = SummaryWriter('tensorboard_log')
try:  # Use try-except to use ctrl+c to stop and save early.
    while training:
        for i, datum in enumerate(data_loader):
            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # Warm up learning rate.
                set_lr(optimizer, (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init)

            # Adjust the learning rate according to the current step.
            while step_index < len(cfg.lr_steps) and step >= cfg.lr_steps[step_index]:
                step_index += 1
                set_lr(optimizer, cfg.lr * (0.1 ** step_index))

            images, box_classes, masks_gt, num_crowds = data_to_device(datum)

            if cuda:
                torch.cuda.synchronize()
            forward_start = time.time()

            predictions = net(images)

            if cuda:
                torch.cuda.synchronize()
            forward_end = time.time()

            losses = criterion(predictions, box_classes, masks_gt, num_crowds)
            losses = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel.
            loss = sum([losses[k] for k in losses])

            optimizer.zero_grad()
            loss.backward()  # Do this to free up vram even if loss is not finite.

            if torch.isfinite(loss).item():
                optimizer.step()

            for k in losses:
                loss_avgs[k].add(losses[k].item())  # Add the loss to the moving average for bookkeeping.

            grad_end = time.time()

            if step > start_step:
                iter_time = grad_end - temp
                batch_time.add(iter_time)
            temp = grad_end

            if step % 10 == 0 and step != start_step:
                cur_lr = optimizer.param_groups[0]['lr']
                seconds = (cfg.max_iter - step) * batch_time.get_avg()
                eta_str = str(datetime.timedelta(seconds=seconds)).split('.')[0]
                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                for k, v in losses.items():
                    writer.add_scalar(f'task/{k}', v, global_step=step)
                writer.add_scalar('total', loss, global_step=step)

                t_forward = forward_end - forward_start
                t_data = iter_time - (grad_end - forward_start)
                time_str = ' T: %.3f | lr: %.2e | t_data: %.3f | t_forward: %.3f | t_total: %.3f | ETA: %s'
                print(('%d |' + (' %s: %.3f |' * len(losses)) + time_str) % tuple(
                    [step] + loss_labels + [total, cur_lr, t_data, t_forward, iter_time, eta_str]), flush=True)

            if args.val_interval > 0 and step % args.val_interval == 0 and step != start_step:
                info = (('iteration: %7d |' + (' %s: %.3f |' * len(losses)) + ' T: %.3f | lr: %.2e')
                        % tuple([step] + loss_labels + [total, cur_lr]))
                table, box_map, mask_map = compute_val_map(net.module)

                writer.add_scalar('box_map', box_map, global_step=step)
                writer.add_scalar('mask_map', mask_map, global_step=step)

                map_tables.append((info, table))
                save_latest(net.module if cuda else net)
                save_best(net.module if cuda else net)

            step += 1
            if step > cfg.max_iter:
                training = False
                break

except KeyboardInterrupt:
    print(f'\nStopped, saving the latest model as \'latest_{cfg.name}_{step}.pth\'.\n')
    save_latest(net.module if cuda else net)
    print_result(map_tables)
    exit()

print(f'Training completed, saving the final model as \'latest_{cfg.name}_{step}.pth\'.\n')
save_latest(net.module if cuda else net)
print_result(map_tables)

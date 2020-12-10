import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data
from tensorboardX import SummaryWriter
import argparse
import datetime
import glob

from utils import timer
from modules.build_yolact import Yolact, NetWithLoss
from modules.multi_loss import Multi_Loss
from config import get_config
from utils.coco import COCODetection, train_collate
from utils.common_utils import save_best, save_latest
from eval import evaluate

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--local_rank', type=int, default=None)
parser.add_argument('--cfg', default='res101_coco', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
parser.add_argument('--img_size', default=550, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=4000, type=int,
                    help='The validation interval during training, pass -1 to disable.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

args = parser.parse_args()
cfg = get_config(args, mode='train')

net = Yolact(cfg)
net.train()

if args.resume == 'latest':
    weight = glob.glob('weights/latest*')[0]
    net.load_weights(weight, cfg.cuda)
    start_step = int(weight.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{weight}\'.\n')
elif args.resume and 'yolact' in args.resume:
    net.load_weights(cfg.weight, cfg.cuda)
    start_step = int(cfg.weight.split('.pth')[0].split('_')[-1])
    print(f'\nResume training with \'{args.resume}\'.\n')
else:
    net.init_weights(cfg.weight)
    print(f'\nTraining from begining, weights initialized with {cfg.weight}.\n')
    start_step = 0

dataset = COCODetection(cfg, mode='train')
optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
criterion = Multi_Loss(cfg)
net = NetWithLoss(net, criterion)

train_sampler = None
main_gpu = False
if cfg.cuda:
    cudnn.benchmark = True
    cudnn.fastest = True
    main_gpu = dist.get_rank() == 0
    num_gpu = dist.get_world_size()

    net = DDP(net.cuda(), [args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
    train_sampler = DistributedSampler(dataset, shuffle=True)

# If encounters OOM error when training on a 11GB memory GPU with batch_size=8, try set pin_memory= False,
# not sure if this helps.
# shuffle must be False if sampler is specified
# data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=cfg.bs_per_gpu // 2, shuffle=(train_sampler is None),
#                               collate_fn=train_collate, pin_memory=True, sampler=train_sampler)
data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=0, shuffle=True,
                              collate_fn=train_collate, pin_memory=True)
step_index = 0
epoch_seed = 0
map_tables = []
training = True
timer.reset()
step = start_step
cfg_name = cfg.__class__.__name__
writer = SummaryWriter('tensorboard_log')

try:  # try-except can shut down all processes after Ctrl + C.
    while training:
        if train_sampler:
            epoch_seed += 1
            train_sampler.set_epoch(epoch_seed)

        for images, targets, masks in data_loader:
            if ((not cfg.cuda) or main_gpu) and step == start_step + 1:
                timer.start()

            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # Warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init

            # Adjust the learning rate according to the current step.
            while step_index < len(cfg.lr_steps) and step >= cfg.lr_steps[step_index]:
                step_index += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * (0.1 ** step_index)

            if cfg.cuda:
                images = images.cuda().detach()
                targets = [ann.cuda().detach() for ann in targets]
                masks = [mask.cuda().detach() for mask in masks]

            with timer.counter('for+loss'):
                loss_b, loss_m, loss_c, loss_s = net(images, targets, masks)

                if cfg.cuda:
                    # use .all_reduce() to get the summed loss from all GPUs
                    all_loss = torch.stack([loss_b, loss_m, loss_c, loss_s], dim=0)
                    dist.all_reduce(all_loss)
                    all_loss_sum = all_loss.sum()

            with timer.counter('backward'):
                loss_total = loss_b + loss_m + loss_c + loss_s
                optimizer.zero_grad()
                loss_total.backward()

            with timer.counter('update'):
                finite_loss = torch.isfinite(all_loss_sum) if cfg.cuda else torch.isfinite(loss_total)

                if finite_loss:
                    optimizer.step()
                else:
                    print(f'Infinite loss, step: {step}')

            time_this = time.time()
            if step > start_step:
                batch_time = time_this - time_last
                timer.add_batch_time(batch_time)
            time_last = time_this

            if step % 20 == 0 and step != start_step:
                if finite_loss and ((not cfg.cuda) or main_gpu):
                    cur_lr = optimizer.param_groups[0]['lr']
                    time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
                    t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                    seconds = (cfg.max_iter - step) * t_t
                    eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                    # Get the mean loss across all GPUS for printing, seems need to call .item(), not sure
                    l_b = all_loss[0].item() / num_gpu if main_gpu else loss_b.item()
                    l_m = all_loss[1].item() / num_gpu if main_gpu else loss_m.item()
                    l_c = all_loss[2].item() / num_gpu if main_gpu else loss_c.item()
                    l_s = all_loss[3].item() / num_gpu if main_gpu else loss_s.item()

                    writer.add_scalar(f'task/box', l_b, global_step=step)
                    writer.add_scalar(f'task/mask', l_m, global_step=step)
                    writer.add_scalar(f'task/class', l_c, global_step=step)
                    writer.add_scalar(f'task/semantic', l_s, global_step=step)
                    writer.add_scalar('total', loss_total, global_step=step)

                    print(f'step: {step} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | '
                          f'l_mask: {l_m:.3f} | l_semantic: {l_s:.3f} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | '
                          f't_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

            if args.val_interval > 0 and step % args.val_interval == 0 and step != start_step:
                if (not cfg.cuda) or main_gpu:
                    net.eval()
                    table, box_row, mask_row = evaluate(net.module.net, cfg, step)
                    map_tables.append(table)
                    net.train()
                    timer.reset()  # training timer and val timer share the same Obj, so reset it to avoid conflict

                    writer.add_scalar('box_map', box_row[1], global_step=step)
                    writer.add_scalar('mask_map', mask_row[1], global_step=step)

                    save_best(net.module.net if cfg.cuda else net, mask_row[1], cfg_name, step)

            if ((not cfg.cuda) or main_gpu) and step != 1 and step % cfg.val_interval == 1:
                timer.start()  # the first iteration after validation should not be included

            step += 1
            if step > cfg.max_iter:
                training = False

                if (not cfg.cuda) or main_gpu:
                    print(f'Training completed.')
                    save_latest(net.module.net if cfg.cuda else net, cfg_name, step)

                    print('\nValidation results during training:\n')
                    for table in map_tables:
                        print(table, '\n')

                break

except KeyboardInterrupt:
    if (not cfg.cuda) or main_gpu:
        save_latest(net.module.net if cfg.cuda else net, cfg_name, step)

        print('\nValidation results during training:\n')
        for table in map_tables:
            print(table, '\n')

from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils import timer
from modules.build_yolact import Yolact
import time
import torch
from modules.multi_loss import Multi_Loss
from data.config import set_cfg, cfg, MEANS
from data.coco import COCODetection
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import datetime
import eval
from data.coco import detection_collate

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--config', default=None, help='The config object to use.')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--resume', default=None, type=str, help='The path of checkpoint file to resume training from.')
parser.add_argument('--lr', '--learning_rate', default=None, type=float, help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float, help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float, help='Weight decay for SGD. Leave as None to read this from the config.')
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) is None:
        setattr(args, name, getattr(cfg, name))

replace('lr')
replace('decay')
replace('momentum')
print(vars(args), '\n')

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


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
                                    augmentation=BaseTransform(MEANS))
        yolact_net.eval()
        print("\nComputing validation mAP...", flush=True)
        table = eval.evaluate(yolact_net, val_dataset, during_training=True)
        yolact_net.train()
        return table

def val_result(map_tables):
    print('Validation results during training:\n')
    for iteration, table in map_tables:
        print(f'iteration: {iteration}')
        print(table, '\n')

def train():
    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            augmentation=SSDAugmentation(MEANS))

    net = Yolact()
    net.train()

    # Don't use the timer during training, there's a race condition with multiple GPUs.
    timer.disable_all()

    if args.resume is not None:
        net.load_weights(args.resume)
        resume_iter = SavePath.from_str(args.resume).iteration
        print(f'\nResume training, start at iteration: {resume_iter}.')
    else:
        print('Initializing weights...')
        net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    criterion = Multi_Loss(num_classes=cfg.num_classes,
                           pos_thre=cfg.positive_iou_threshold,  # 0.5
                           neg_thre=cfg.negative_iou_threshold,  # 0.4
                           negpos_ratio=3)

    if cuda:
        cudnn.benchmark = True
        net = nn.DataParallel(net).cuda()
        criterion = nn.DataParallel(criterion).cuda()

    iteration = resume_iter if args.resume else 0
    epoch_size = len(dataset) // args.batch_size
    num_epochs = cfg.max_iter // epoch_size + 1
    step_index = 0

    data_loader = data.DataLoader(dataset,
                                  args.batch_size,
                                  num_workers=4,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    time_avg = MovingAverage()

    loss_types = ['B', 'C', 'M', 'S']
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }
    map_tables = []

    print('Begin training!\n')
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            for datum in data_loader:
                torch.cuda.synchronize()
                train_start = time.time()
                iteration += 1

                # Warm up learning rate
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    # lr' = lr * 0.1 ^ step_index
                    set_lr(optimizer, args.lr * (0.1 ** step_index))

                images, box_classes, masks_gt, num_crowds = data_to_device(datum)

                predictions = net(images)
                optimizer.zero_grad()

                losses = criterion(predictions, box_classes, masks_gt, num_crowds)
                losses = {k: v.mean() for k, v in losses.items()}    # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])

                loss.backward()    # Do this to free up vram even if loss is not finite

                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_lr = optimizer.param_groups[0]['lr']

                torch.cuda.synchronize()
                train_end = time.time()
                elapsed = train_end - train_start
                time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d |' + (' %s: %.3f |' * len(losses)) + ' T: %.3f | lr: %.5f | ETA: %s | current: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, cur_lr, eta_str, elapsed]), flush=True)

                if iteration == cfg.max_iter:
                    break

            net.module.save_weights(f'\nweights/{cfg.name}_{epoch}_{iteration}.pth')
            print(f'Model saved in weights: {cfg.name}_{epoch}_{iteration}.pth')

            table = compute_val_map(net.module)
            map_tables.append((iteration, table))

    except KeyboardInterrupt:
        print(f'\nKeyboardInterrupt, saving network at epoch: {epoch}, iteration: {iteration}.\n\n')
        net.module.save_weights(f'weights/{cfg.name}_{epoch}_{iteration}.pth')

        val_result(map_tables)
        exit()

    print(f'Training completed, saving network at epoch: {epoch}, iteration: {iteration}.\n')
    net.module.save_weights(f'weights/{cfg.name}_{epoch}_{iteration}.pth')

    val_result(map_tables)


if __name__ == '__main__':
    train()

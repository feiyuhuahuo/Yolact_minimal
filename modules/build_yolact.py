import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from math import sqrt
from typing import List
from data.config import cfg
from modules.backbone import construct_backbone
from utils import timer
torch.cuda.current_device()

class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

class InterpolateModule(nn.Module):
    """
    A module version of F.interpolate.
    """
    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)

def make_net(in_channels, cfg_net, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network. Used by protonet and extra head.
    return:
        network, out_channels
    """
    def make_layer(layer_cfg):
        nonlocal in_channels

        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list

        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])

            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat all the component layer lists
    net = sum([make_layer(x) for x in cfg_net], [])    # x: (256, 3, {'padding': 1})

    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*net), in_channels


class PredictionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.coef_dim    = cfg.coef_dim
        self.num_priors  = len(cfg.backbone.aspect_ratios)

        if cfg.extra_head_net is None:
            out_channels = in_channels
        else:
            self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.coef_dim, kernel_size=3, padding=1)

    def forward(self, x):
        if cfg.extra_head_net is not None:
            x = self.upfeature(x)

        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.coef_dim)
        coef = torch.tanh(coef)
        
        return {'box': bbox, 'class': conf, 'coef': coef}

class FPN(nn.Module):
    """
    Implements a general version of the FPN introduced in https://arxiv.org/pdf/1612.03144.pdf

    - num_features (int): The number of output features in the fpn layers.
    - interpolation_mode (str): The mode to pass to F.interpolate.
    - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                            These extra layers are downsampled from the last selected layer.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.interpolation_mode  = cfg.fpn.interpolation_mode    # 'bilinear'
        self.num_downsample      = cfg.fpn.num_downsample        # 2
        self.conv_downsample = cfg.fpn.use_conv_downsample   # True
        self.num_features = cfg.fpn.num_features
        self.in_channels = in_channels
        self.padding = cfg.fpn.pad  # for backwards compatability

        self.lat_layers  = nn.ModuleList([nn.Conv2d(x, self.num_features, kernel_size=1) for x in reversed(self.in_channels)])
        # ModuleList((0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))

        self.pred_layers = nn.ModuleList([nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=self.padding)
                                          for _ in self.in_channels])
        # ModuleList((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #            (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        if self.conv_downsample:
            self.downsample_layers = nn.ModuleList([nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, stride=2)
                                                    for _ in range(self.num_downsample)])

        # ModuleList((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  用于下采样
        #            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

    def forward(self, convouts: List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)    # convouts: C3, C4, C5

        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()

                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)

            x = x + lat_layer(convouts[j])
            out[j] = x

        # This second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for i in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out

class Anchors:
    def __init__(self):
        self.conv_h = None
        self.conv_w = None
        self.aspect_ratios = cfg.backbone.aspect_ratios
        self.last_conv_size = None

    def make_anchors(self, conv_h, conv_w, scale):
        if self.last_conv_size != (conv_h, conv_w):
            prior_data = []

            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(conv_h), range(conv_w)):

                # + 0.5 because priors are in center
                x = (i + 0.5) / conv_w
                y = (j + 0.5) / conv_h

                for ar in self.aspect_ratios:
                    ar = sqrt(ar)
                    w = scale * ar / cfg.max_size
                    h = scale / ar / cfg.max_size

                    # This is for backward compatability with a bug where I made everything square by accident
                    if cfg.backbone.use_square_anchors:  # True
                        h = w

                    prior_data += [x, y, w, h]

            self.priors = torch.Tensor(prior_data).view(-1, 4).cuda()
            self.last_conv_size = (conv_h, conv_w)

        return self.priors


class Yolact(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = construct_backbone(cfg.backbone)
        # import torchsummary
        # torchsummary.summary(self.backbone, (3, 550, 550))

        self.scales = cfg.backbone.scales
        self.aspect_scales = cfg.backbone.aspect_ratios

        if cfg.freeze_bn:
            self.freeze_bn()

        in_channels = cfg.fpn.num_features    # 256

        self.proto_net, cfg.coef_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)
        '''  
        self.proto_net:
        Sequential((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (1): ReLU(inplace)
                   (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (3): ReLU(inplace)
                   (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (5): ReLU(inplace)
                   (6): InterpolateModule()
                   (7): ReLU(inplace)
                   (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                   (9): ReLU(inplace)
                   (10): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1)))
        cfg.coef_dim: 32
        '''

        self.fpn = FPN([512, 1024, 2048])
        self.selected_layers = [0, 1, 2, 3, 4]
        # create a ModuleList to match with the original pre-trained weights (original model state_dict)
        self.prediction_layers = nn.ModuleList()
        self.prediction_layers.append(PredictionModule(in_channels))

        '''  
        self.prediction_layers:
        ModuleList(
          (0): PredictionModule((upfeature): Sequential((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                                        (1): ReLU(inplace))
                                (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                (mask_layer): Conv2d(256, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
        '''
        self.anchor_list = [Anchors() for _ in self.selected_layers]

        if cfg.use_semantic_segmentation_loss:    # True
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes-1, kernel_size=1)    # Conv2d(256, 80, kernel_size=1)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        state_dict = torch.load(path)

        for key in list(state_dict.keys()):
            # 'fpn.downsample_layers.2.weight' and 'fpn.downsample_layers.2.bias' in the pretrained .pth are redundant, remove them
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        # Initialize the rest conv layers with xavier
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                # torch.manual_seed(10)
                # torch.cuda.manual_seed(10)
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def forward(self, x):
        """ x: [batch_size, 3, img_h, img_w] """
        with timer.env('backbone'):
            outs = self.backbone(x)

        with timer.env('fpn'):
            outs = [outs[i] for i in cfg.backbone.selected_layers]
            outs = self.fpn(outs)

        '''
        outs:
        (2, 3, 550, 550) -> backbone -> (2, 256, 138, 138) -> fpn -> [2, 256, 69, 69] P3
                                        (2, 512, 69, 69)             [2, 256, 35, 35] P4
                                        (2, 1024, 35, 35)            [2, 256, 18, 18] P5
                                        (2, 2048, 18, 18)            [2, 256, 9, 9]   P6
                                                                     [2, 256, 5, 5]   P7
        '''
        with timer.env('proto'):
            # outs[0]: [2, 256, 69, 69], the feature map from P3
            proto_out = self.proto_net(outs[0])    # proto_out: [2, 32, 138, 138]
            proto_out = F.relu(proto_out, inplace=True)
            proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        with timer.env('pred_heads'):
            predictions = {'box': [], 'class': [], 'coef': [], 'priors': []}

            for i in self.selected_layers:    # self.selected_layers [0, 1, 2, 3, 4]
                p = self.prediction_layers[0](outs[i])

                for k, v in p.items():
                    predictions[k].append(v)

                conv_h, conv_w = outs[i].size(2), outs[i].size(3)
                anchors = self.anchor_list[i].make_anchors(conv_h, conv_w, self.scales[i])
                predictions['priors'].append(anchors)

        for k, v in predictions.items():
            predictions[k] = torch.cat(v, -2)

        predictions['proto'] = proto_out

        if self.training:
            if cfg.use_semantic_segmentation_loss:   # True
                predictions['segm'] = self.semantic_seg_conv(outs[0])


            return predictions

        else:
            predictions['class'] = F.softmax(predictions['class'], -1)
            return predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg, mask_proto_net, extra_head_net
from modules.backbone import construct_backbone
from utils.box_utils import make_anchors
from utils import timer


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
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
    def make_layer(layer_cfg):
        nonlocal in_channels

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
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False,
                                              **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat all the component layer lists
    net = sum([make_layer(x) for x in cfg_net], [])  # x: (256, 3, {'padding': 1})

    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*net), in_channels


class PredictionModule(nn.Module):
    def __init__(self, in_channels, coef_dim):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim
        self.num_priors = len(cfg.aspect_ratios)

        self.upfeature, out_channels = make_net(in_channels, extra_head_net)
        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.coef_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        coef = torch.tanh(coef)

        return {'box': bbox, 'class': conf, 'coef': coef}


class FPN(nn.Module):
    """
    The FPN here is slightly different from the FPN introduced in https://arxiv.org/pdf/1612.03144.pdf.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.num_downsample = 2
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in reversed(self.in_channels)])
        # ModuleList((0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #            (2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)))

        self.pred_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in self.in_channels])
        # ModuleList((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #            (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.downsample_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
                                                for _ in range(self.num_downsample)])
        # ModuleList((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

    def forward(self, backbone_outs):
        out = []
        x = torch.zeros(1, device=backbone_outs[0].device)
        for i in range(len(backbone_outs)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(backbone_outs)  # convouts: C3, C4, C5

        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(backbone_outs) - 1:
                _, _, h, w = backbone_outs[j].size()
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            x = x + lat_layer(backbone_outs[j])

            out[j] = x

        j = len(backbone_outs)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        for layer in self.downsample_layers:
            out.append(layer(out[-1]))

        return out


class Yolact(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchors = []
        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        self.proto_net, coef_dim = make_net(256, mask_proto_net, include_last_relu=False)
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
        '''

        self.fpn = FPN([512, 1024, 2048])
        self.selected_layers = [0, 1, 2, 3, 4]
        # create a ModuleList to match with the original pre-trained weights (original model state_dict)
        self.prediction_layers = nn.ModuleList()
        self.prediction_layers.append(PredictionModule(in_channels=256, coef_dim=coef_dim))
        '''  
        self.prediction_layers:
        ModuleList(
          (0): PredictionModule((upfeature): Sequential((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                                        (1): ReLU(inplace))
                                (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                (mask_layer): Conv2d(256, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
        '''

        if cfg.train_semantic:  # True
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1)

    def load_weights(self, path, cuda):
        if cuda:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')

        for key in list(state_dict.keys()):
            # 'fpn.downsample_layers.2.weight' and 'fpn.downsample_layers.2.bias'
            # in the pretrained .pth are redundant, remove them
            if key.startswith('fpn.downsample_layers.'):
                if int(key.split('.')[2]) >= 2:
                    del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)
        # Initialize the rest conv layers with xavier
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
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
        with timer.env('backbone'):
            outs = self.backbone(x)

        with timer.env('fpn'):
            outs = [outs[i] for i in cfg.backbone.selected_layers]
            outs = self.fpn(outs)

            '''
            outs:
            (n, 3, 550, 550) -> backbone -> (n, 256, 138, 138) -> fpn -> (n, 256, 69, 69) P3
                                            (n, 512, 69, 69)             (n, 256, 35, 35) P4
                                            (n, 1024, 35, 35)            (n, 256, 18, 18) P5
                                            (n, 2048, 18, 18)            (n, 256, 9, 9)   P6
                                                                         (n, 256, 5, 5)   P7
            '''
        if isinstance(self.anchors, list):
            for i, shape in enumerate([list(aa.shape) for aa in outs]):
                self.anchors += make_anchors(shape[2], shape[3], cfg.scales[i])
            self.anchors = torch.Tensor(self.anchors).view(-1, 4)

        with timer.env('proto'):
            # outs[0]: [2, 256, 69, 69], the feature map from P3
            proto_out = self.proto_net(outs[0])  # proto_out: (n, 32, 138, 138)
            proto_out = F.relu(proto_out, inplace=True)
            proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        with timer.env('pred_heads'):
            predictions = {'box': [], 'class': [], 'coef': []}

            for i in self.selected_layers:  # self.selected_layers [0, 1, 2, 3, 4]
                p = self.prediction_layers[0](outs[i])

                for k, v in p.items():
                    predictions[k].append(v)

        for k, v in predictions.items():
            predictions[k] = torch.cat(v, -2)

        predictions['proto'] = proto_out
        predictions['anchors'] = self.anchors

        if self.training:
            if cfg.train_semantic:  # True
                predictions['segm'] = self.semantic_seg_conv(outs[0])
            return predictions

        else:
            predictions['class'] = F.softmax(predictions['class'], -1)
            return predictions

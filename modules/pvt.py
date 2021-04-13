import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor

        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qq = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        kk, vv = kv[0], kv[1]

        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ vv).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.H, self.W = self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h_out, w_out = x.shape[2:4]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (h_out, w_out)


class PyramidVisionTransformer(nn.Module):
    # tiny depth (2, 2, 2, 2)
    def __init__(self, img_size=224, patch_size=4, in_ch=3, embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(8, 8, 4, 4), drop_path_rate=0.1, depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.strides = (1, 4, 8, 16)

        for i in range(4):
            patch_embed = PatchEmbed(img_size=img_size // self.strides[i], patch_size=patch_size if i == 0 else 2,
                                     in_ch=in_ch if i == 0 else embed_dims[i - 1], embed_dim=embed_dims[i])

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'pos_embed{i + 1}', nn.Parameter(
                torch.zeros(1, patch_embed.num_patches + 1 if i == 3 else patch_embed.num_patches, embed_dims[i])))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for k in range(4):
            encoder = nn.ModuleList([Block(dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k],
                                           drop_path=dpr[cur + i], sr_ratio=sr_ratios[k]) for i in range(depths[k])])

            setattr(self, f'block{k + 1}', encoder)
            cur += depths[k]

    def init_backbone(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print(f'\nBackbone is initiated with {path}.\n')

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                                 size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, x):
        outs = []
        B = x.shape[0]

        for i in range(4):  # for stage i
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            pos_embed = getattr(self, f'pos_embed{i + 1}')
            encoder = getattr(self, f'block{i + 1}')

            x, (H, W) = patch_embed(x)
            pos_embed = self._get_pos_embed(pos_embed[:, 1:] if i == 3 else pos_embed, patch_embed, H, W)
            x = x + pos_embed

            for blk in encoder:
                x = blk(x, H, W)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


if __name__ == '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)

    net = PyramidVisionTransformer(img_size=544).cuda()
    net.init_backbone('../weights/pvt_small_no_pos_embed.pth')

    inn = torch.ones((2, 3, 544, 544)).cuda()

    gg = torch.cuda.list_gpu_processes()

    print(gg)
    print(torch.cuda.memory_allocated(device=0) / 1024 / 1024)
    bb = net(inn)
    for gg in bb:
        print(gg.sum())

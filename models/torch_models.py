import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, ks=3):
    pad = ks//2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, ks, padding=pad, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, ks, padding=pad, bias=False),
        nn.ReLU(inplace=True),
    )

class UNetSmall(nn.Module):
    def __init__(self, channels, ks=3, use_skip=True):
        super().__init__()
        c1, c2, c3 = channels
        self.down1 = conv_block(1, c1, ks)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(c1, c2, ks)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(c2, c3, ks)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c3 if use_skip else c2, c2, ks)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c2 if use_skip else c1, c1, ks)
        self.out = nn.Conv2d(c1, 1, 1)
        self.use_skip = use_skip

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        bn = self.bottleneck(p2)
        u2 = self.up2(bn)
        if self.use_skip:
            u2 = torch.cat([u2, d2], dim=1)
        d2r = self.dec2(u2)
        u1 = self.up1(d2r)
        if self.use_skip:
            u1 = torch.cat([u1, d1], dim=1)
        d1r = self.dec1(u1)
        return self.out(d1r)

class FractalUNetSmall(nn.Module):
    def __init__(self, channels, ks=3, branching_factor=2, levels=3, use_skip=True):
        super().__init__()
        c1, c2, c3 = channels
        self.l1_paths = nn.ModuleList([conv_block(1, c1, ks) for _ in range(max(1, branching_factor))])
        self.pool1 = nn.MaxPool2d(2)
        self.l2_paths = nn.ModuleList([conv_block(c1, c2, ks) for _ in range(max(1, branching_factor))])
        self.pool2 = nn.MaxPool2d(2)
        self.bn_paths = nn.ModuleList([conv_block(c2, c3, ks) for _ in range(max(1, levels))])
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c3 if use_skip else c2, c2, ks)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c2 if use_skip else c1, c1, ks)
        self.out = nn.Conv2d(c1, 1, 1)
        self.use_skip = use_skip

    def forward(self, x):
        d1 = torch.stack([path(x) for path in self.l1_paths], dim=0).mean(dim=0)
        p1 = self.pool1(d1)
        d2 = torch.stack([path(p1) for path in self.l2_paths], dim=0).mean(dim=0)
        p2 = self.pool2(d2)
        bn = 0
        for path in self.bn_paths:
            bn = bn + path(p2)
        bn = bn / len(self.bn_paths)
        u2 = self.up2(bn)
        if self.use_skip:
            u2 = torch.cat([u2, d2], dim=1)
        d2r = self.dec2(u2)
        u1 = self.up1(d2r)
        if self.use_skip:
            u1 = torch.cat([u1, d1], dim=1)
        d1r = self.dec1(u1)
        return self.out(d1r)

def build_from_manifest(manifest: dict):
    t = manifest.get("type")
    ch = manifest.get("channels", [32,64,128])
    ks = int(manifest.get("kernel_size", 3))
    use_skip = bool(manifest.get("use_skip", True))
    if t == "unet":
        return UNetSmall(channels=ch, ks=ks, use_skip=use_skip)
    elif t == "fractal_unet":
        bf = int(manifest.get("branching_factor", 2))
        lv = int(manifest.get("levels", len(ch)))
        return FractalUNetSmall(channels=ch, ks=ks, branching_factor=bf, levels=lv, use_skip=use_skip)
    else:
        raise ValueError(f"Unknown model type: {t}")

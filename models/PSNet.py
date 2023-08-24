import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from models import model_utils
from models import NAFNet


class Residual(nn.Module):
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        hdim = int(outs / 2)
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ins, hdim, 1),
            nn.BatchNorm2d(hdim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hdim, hdim, 3, 1, 1),
            nn.BatchNorm2d(hdim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hdim, outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class ResBlock(nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.1, inplace=True)
            # nn.GELU()
        )

    def forward(self, x):
        out = x + self.layers(x)
        return out


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction, bias=False, bn=False):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.ReflectionPad2d(1))
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(nn.LeakyReLU(0.1, inplace=True))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = 1

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = nn.Sequential(*([NAFNet.NAFBlock(64)] * 4))
        self.conv3 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv4 = nn.Sequential(*([NAFNet.NAFBlock(128)] * 2))
        self.conv5 = model_utils.conv(batchNorm, 128, 256, k=3, stride=1, pad=1)
        # self.conv4 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        # self.conv4 = nn.Sequential(*([RCAB(128, 16, bias=False, bn=False)] * 2))
        # self.conv5 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        # self.conv6 = ResBlock(256)
        # self.conv7 = model_utils.deconv(256, 128)
        self.conv6 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        out_feat = self.conv6(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 256, 128, k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        # self.deconv2 = model_utils.upsampling(128, 64)
        # self.deconv3 = model_utils.pixel_shuffle(64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=0, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1:  # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feats = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)  # 将光照方向与图像拼接在一起
            feat, shape = self.extractor(net_in)
            feats.append(feat)
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        normal = self.regressor(feat_fused, shape)
        return normal


if __name__ == '__main__':
    x = torch.randn((1, 3, 128, 128))
    # conv1 = model_utils.conv(batchNorm=False, cin=3, cout=64, k=3, stride=1, pad=1)
    # conv2 = NAFNet.NAFBlock(c=64, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.)
    # conv3 = model_utils.conv(batchNorm=False, cin=64, cout=128, k=3, stride=2, pad=1)
    # conv4 = NAFNet.NAFBlock(c=128, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.)
    #
    # out1 = conv1(x)
    # print(f'out1 shape is {out1.shape}')
    # out2 = conv2(out1)
    # print(f'out2 shape is {out2.shape}')
    # out3 = conv3(out2)
    # print(f'out3 shape is {out3.shape}')
    # out4 = conv4(out3)
    # print(f'out4 shape is {out4.shape}')
    extractor = FeatExtractor(batchNorm=False, c_in=3, other={})
    feat, shape=extractor(x)
    print(f'shape is {shape}')

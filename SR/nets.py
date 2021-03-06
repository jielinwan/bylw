import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.spectral_norm as SpectralNorm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Residual(nn.Module):
    """
    from cydiachen's implementation
    (https://github.com/cydiachen/FSRNET_pytorch)
    """
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        hdim = int(outs/2)
        self.convBlock = nn.Sequential(
            #nn.BatchNorm2d(ins),
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.ReLU(True),
            nn.Conv2d(ins, hdim, 1),
            #nn.BatchNorm2d(hdim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.ReLU(True),
            nn.Conv2d(hdim, hdim, 3, 1, 1),

            #nn.BatchNorm2d(hdim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.ReLU(True),
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


class HourGlassBlock(nn.Module):
    """
    from cydiachen's implementation
    (https://github.com/cydiachen/FSRNET_pytorch)
    """
    def __init__(self, dim, n):
        super(HourGlassBlock, self).__init__()
        self._dim = dim
        self._n = n
        self._init_layers(self._dim, self._n)

    def _init_layers(self, dim, n):
        setattr(self, 'res'+str(n)+'_1', Residual(dim, dim))
        setattr(self, 'pool'+str(n)+'_1', nn.MaxPool2d(2,2))
        setattr(self, 'res'+str(n)+'_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n-1)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self,'res'+str(n)+'_3', Residual(dim, dim))
        setattr(self,'unsample'+str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res'+str(n)+'_1')(up1)
        low1 = eval('self.pool'+str(n)+'_1')(x)
        low1 = eval('self.res'+str(n)+'_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n-1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.'+'res'+str(n)+'_3')(low3)
        up2 = eval('self.'+'unsample'+str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)


class ResBlock(nn.Module):

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(dim),
            #nn.ReLU(True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.layers(x)
        return out

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights(
        #    [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x



class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x



class CoarseSRNetwork(nn.Module):

    def __init__(self):
        super(CoarseSRNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            #nn.ReLU(True),
        )
        #self.res_blocks = nn.Sequential(*([ResBlock(64)] * 3))
        self.rrdb_blocks = nn.Sequential(*([RRDB(64)] * 9))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.Tanh(),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
           # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        #out = self.res_blocks(out_conv1)
        out = self.rrdb_blocks(out_conv1)
        out=self.conv2(out)
        out=out+out_conv1
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.conv3(out)
        return out


class FineSREncoder(nn.Module):

    def __init__(self):
        super(FineSREncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(*([ResBlock(64)] * 7 ))
        #self.rrdb_blocks = nn.Sequential(*([RRDB(64)] * 3))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.LeakyReLU(negative_slope=0.2,inplace=True)
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out=self.res_blocks(out)
        #out = self.rrdb_blocks(out)
        out = self.conv2(out)
        return out


class PriorEstimationNetwork(nn.Module):

    def __init__(self):
        super(PriorEstimationNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(
            Residual(64, 128),
            ResBlock(128),
            ResBlock(128),
        )
        self.hg_blocks = nn.Sequential(
            HourGlassBlock(128, 3),
            HourGlassBlock(128, 3),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.hg_blocks(out)
        return out


class FineSRDecoder(nn.Module):

    def __init__(self):
        super(FineSRDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(75, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.ReLU(True),
        )

        self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.deconv1=nn.Sequential(
        #    nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1, output_padding=1,bias=False),
        #    nn.BatchNorm2d(64),
        #    nn.ReLU(True),
        #)

        self.res_blocks = nn.Sequential(*([ResBlock(64)] * 3))
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        out = self.conv1(x)
        out=self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        #out=self.deconv1(out)
        #out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.res_blocks(out)
        out = self.conv2(out)
        return out


class FSRNet(nn.Module):

    def __init__(self, hmaps_ch, pmaps_ch):
        super(FSRNet, self).__init__()
        self.hmaps_ch = hmaps_ch
        self.pmaps_ch = pmaps_ch

        self.csr_net = CoarseSRNetwork()
        self.fsr_enc = FineSREncoder()
        self.pre_net = PriorEstimationNetwork()
        self.fsr_dec = FineSRDecoder()

        # 1x1 conv for hmaps & pmaps
        self.prior_conv1 = None
        self.prior_conv2 = None
        if self.hmaps_ch > 0:
            self.prior_conv1 = nn.Conv2d(128, self.hmaps_ch, kernel_size=1) # hmaps (landmark pts)
        if self.pmaps_ch > 0:
            self.prior_conv2 = nn.Conv2d(128, self.pmaps_ch, kernel_size=1) # pmaps (parsing maps)

    def forward(self, x):
        y_c = self.csr_net(x)

        f = self.fsr_enc(y_c)

        p = self.pre_net(y_c)

        # 1x1 conv for hmaps & pmaps
        b1 = (self.prior_conv1 is not None)
        b2 = (self.prior_conv2 is not None)
        if b1 and b2:
            hmaps = self.prior_conv1(p)
            pmaps = self.prior_conv2(p)
            prs = torch.cat((hmaps, pmaps), 1)
        elif b1:
            prs = self.prior_conv1(p)
        elif b2:
            prs = self.prior_conv2(p)

        concat = torch.cat((f, prs), 1)
        out = self.fsr_dec(concat)

        return y_c, prs, out


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg16_model = torchvision.models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg16_model.features.children())[:30])

        '''conv_5_3_layer = 14  # layer index of relu5_3 in vgg-16
        cnn = torchvision.models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        self.model = nn.Sequential()
        self.model = self.model.cuda()
        for i, layer in enumerate(list(cnn)):
            self.model.add_module(str(i), layer)
            if i == conv_5_3_layer:
                break'''

    def forward(self, x):
        return self.feature_extractor(x)
        #return self.model(x)

class Discriminator(nn.Module):
    """
    from eriklindernoren's SRGAN implementation
    (https://github.com/eriklindernoren/PyTorch-GAN)
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(SpectralNorm(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)))
            #if not first_block:
            #    layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(SpectralNorm(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(SpectralNorm(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)))
            #layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(SpectralNorm(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)))
        
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':

    # x = torch.randn((1, 3, 128, 128))
    # net = FSRNet(hmaps_ch=0, pmaps_ch=11)
    # y_c, prs, out = net(x)
    # print(' input:', x.size())
    # print('   y_c:', y_c.size())
    # print(' prior:', prs.size())
    # print('     y:', out.size())

    import numpy as np
    x = torch.randn((1, 3, 128, 128))
    G = FSRNet(hmaps_ch=0, pmaps_ch=11)
    D = Discriminator(input_shape=(3, 128, 128))
    F = FeatureExtractor()
    criterion = nn.MSELoss()
    
    y_c, prs, out = G(x)
    print(' input:', x.size())
    print('   y_c:', y_c.size())
    print(' prior:', prs.size())
    print('     y:', out.size())
    print()

    valid = torch.from_numpy(np.ones((x.size(0), *D.output_shape)))
    fake = torch.from_numpy(np.zeros((x.size(0), *D.output_shape)))
    valid.requires_grad = False
    fake.requires_grad = False

    disc = D(out)
    criterion(F(out), F(x).detach())
    print(disc.size())

import torch
import torch.nn as nn
from math import log
from torch.autograd import Variable
from torch.nn import Upsample
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

# ############## G networks ################################################
class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

def jointConv(in_channels1, in_channels2, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels1 + in_channels2, out_channels = out_channels * 2,\
            kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels * 2),
        GLU()
    )

def RGBLayer(in_features):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_features, out_channels = 3, \
            kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.Tanh()
    )

def BinaryLayer(in_features):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_features, out_channels = 1, \
            kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.Sigmoid()
    )

class upBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(upBlock, self).__init__()

        self.up_sampler = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer1 = nn.Sequential(
            conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            GLU()
        )

    def forward(self, x):
        y = self.up_sampler(x)
        y = self.conv_layer1(y)
        return y

class ConditionAugmentationModule(nn.Module):
    def __init__(self, t_dim, s_dim):
        super(ConditionAugmentationModule, self).__init__()
        self.t_dim = t_dim
        self.s_dim = s_dim
        self.fc = nn.Linear(self.t_dim, self.s_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.s_dim]
        logvar = x[:, self.s_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar, eps = None):
        std = logvar.mul(0.5).exp_()
        if eps is None:
            # eps = torch.FloatTensor(std.size()).normal_()
            eps = torch.cuda.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding, eps = None):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar, eps)
        return c_code, mu, logvar

class AttentionModule(nn.Module):
    def __init__(self, in_features, t_dim):
        super(AttentionModule, self).__init__()
        self.conv_context = conv1x1(t_dim, in_features)
        self.sm = nn.Softmax()

    def forward(self, input, context, mask):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)
        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        # batch_size x sourceL --> batch_size*queryL x sourceL
        mask = mask.repeat(queryL, 1)
        attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()
        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

class WordLevelCombinedUpsampler(nn.Module):
    def __init__(self, in_features, t_dim):
        super(WordLevelCombinedUpsampler, self).__init__()
        self.att = AttentionModule(in_features = in_features, t_dim = t_dim)
        self.residual = WordLevelCombinedUpsampler.__make_layer(in_features = in_features * 2)
        self.upsample = upBlock(in_features * 2, in_features)

    @staticmethod
    def __make_layer(in_features, num_residual = 2):
        layers = []
        for i in range(num_residual):
            layers.append(ResBlock(in_features))
        return nn.Sequential(*layers)

    def forward(self, feature, word_embs, mask):
        c_code, _ = self.att(feature, word_embs, mask)
        h_c_code = torch.cat((feature, c_code), 1)
        out_code = self.residual(h_c_code)
        out_code = self.upsample(out_code)

        return out_code

class CooperationModule(nn.Module):
    def __init__(self, in_channels1, in_channels2, bypass_channels1to2, bypass_channels2to1):
        super(CooperationModule, self).__init__()
        
        self.bypass1to2 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels1, out_channels = bypass_channels1to2 * 2, \
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bypass_channels1to2 * 2),
            GLU()
        )
        self.bypass2to1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels2, out_channels = bypass_channels2to1 * 2, \
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bypass_channels2to1 * 2),
            GLU()
        )

    def forward(self, x1, x2):
        x1_bypass = self.bypass2to1(x2)
        x2_bypass = self.bypass1to2(x1)

        y1 = torch.cat((x1_bypass, x1), 1)
        y2 = torch.cat((x2_bypass, x2), 1)

        return y1, y2

class CooperationUpsampleModule(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(CooperationUpsampleModule, self).__init__()

        self.bypass_channels1to2 = in_channels2 // 4
        self.bypass_channels2to1 = in_channels1 // 4

        self.cooperation_model = CooperationModule(in_channels1, in_channels2, self.bypass_channels1to2, self.bypass_channels2to1)
        self.upsampler1 = upBlock(in_channels1 + self.bypass_channels2to1, in_channels1 // 2)
        self.upsampler2 = upBlock(in_channels2 + self.bypass_channels1to2, in_channels2 // 2)
        #self.upsampler1 = upBlock(in_channels1, in_channels1 // 2)
        #self.upsampler2 = upBlock(in_channels2, in_channels2 // 2)
        

    def forward(self, x1, x2):
        x1_c, x2_c = self.cooperation_model(x1, x2)
        y1 = self.upsampler1(x1_c)
        y2 = self.upsampler2(x2_c)
        #y1 = self.upsampler1(x1)
        #y2 = self.upsampler2(x2)

        return y1, y2

class DirectUpsampleModule(nn.Module):

    def __init__(self, in_channels):
        super(DirectUpsampleModule, self).__init__()
        self.up_sampler = upBlock(in_planes = in_channels, out_planes = in_channels // 2)

    def forward(self, x):
        return self.up_sampler(x)

class NoiseInModule(nn.Module):

    def __init__(self, in_features, noise_channels, res_num = 1):
        super(NoiseInModule, self).__init__()
        self.noise_channels = noise_channels
        self.joint_conv = jointConv(in_channels1 = in_features, in_channels2 = noise_channels, out_channels = in_features)
        
        self.res_block = []
        for i in range(res_num):
            self.res_block.append(ResBlock(in_features))
        self.res_block = nn.Sequential(*self.res_block)

    def forward(self, features, noise = None):
        if noise is None:
            noise = torch.randn(features.size(0), self.noise_channels, features.size(2), features.size(3), device=features.device, dtype=features.dtype)
        joint_code = self.joint_conv(torch.cat((features, noise), 1))
        out_code = self.res_block(joint_code)
        return out_code

class MultiStageNoiseInModule(nn.Module):
    def __init__(self, in_features, stage_num):
        super(MultiStageNoiseInModule, self).__init__()

        self.stage_num = stage_num
        self.noise_in_modules = nn.ModuleList(
            NoiseInModule(in_features, in_features // 2) for i in range(stage_num)
        )
        self.rgb_layers = nn.ModuleList(
            # additional RGB layer applied to the given no-noise-adapted features
            RGBLayer(in_features) for i in range(self.stage_num + 1)
        )
        self.msk_layers = nn.ModuleList(
            # additional Binary layer applied to the given no-noise-adapted features
            BinaryLayer(in_features) for i in range(self.stage_num + 1)
        )
        self.combined_conv = \
            nn.Conv2d(in_channels = in_features * (stage_num + 1), out_channels = in_features, 
                kernel_size = 1, stride = 1, padding = 0, bias = False)

    def forward(self, ob_features):
        # original ob_features, no noise adapted
        obs     = []
        masks   = []

        ob      = self.rgb_layers[0](ob_features)
        ob_mask = self.msk_layers[0](ob_features)
        
        obs.append(ob)
        masks.append(ob_mask)
        ob      = ob * ob_mask
        combined_ob_features = ob_features

        # multi-stage noise in
        for i in range(1, self.stage_num + 1):
            ob_features = self.noise_in_modules[i-1](ob_features) # noise in
            obi   = self.rgb_layers[i](ob_features)
            maski = self.msk_layers[i](ob_features)
            ob    = obi * maski + ob * (1 - maski)
            ob_mask = 1 - (1 - ob_mask) * (1 - maski) # similiar to ob_mask = ob_mask | maski
            combined_ob_features = torch.cat((combined_ob_features, ob_features), 1)
            
            obs.append(obi)
            masks.append(maski)

        combined_ob_features = self.combined_conv(combined_ob_features)
        #combined_ob_features = ob_features
        return combined_ob_features, ob, ob_mask, obs, masks

class Generator(nn.Module):
    def __init__(self, z_dim, t_dim, s_dim = 256, resolution = 128, noise_stage_num = 1, base_features = 1024):
        super(Generator, self).__init__()

        assert resolution in (64, 128, 256), 'Resolution should be 64, 128 or 256'
        
        self.noise_stage_num = noise_stage_num
        self.base_features = base_features
        self.progressive_num = int(log(resolution // 64, 2)) + 1
        
        self.ca = ConditionAugmentationModule(t_dim = t_dim, s_dim = s_dim)

        self.bg_dropout = nn.Dropout(0.5)
        self.bg_fc = nn.Sequential(
            nn.Linear(in_features = z_dim + s_dim, out_features = base_features * 4 * 4 * 2, bias = False),
            nn.BatchNorm1d(base_features * 4 * 4 * 2),
            GLU(),
        ) # latent code -> 4x4

        self.ob_fc = nn.Sequential(
            nn.Linear(in_features = z_dim + s_dim, out_features = base_features * 4 * 4 * 2, bias = False),
            nn.BatchNorm1d(base_features * 4 * 4 * 2),
            GLU()
        ) # latent code -> 4x4

        self.init_co_upsamplers = nn.ModuleList([
            CooperationUpsampleModule(base_features,      base_features),       #   8x8
            CooperationUpsampleModule(base_features // 2, base_features // 2),  # 16x16
            CooperationUpsampleModule(base_features // 4, base_features // 4),  # 32x32
            CooperationUpsampleModule(base_features // 8, base_features // 8)   # 64x64
        ])
        
        init_features = base_features // 16

        # bg_pro_upsampler doesn't apply a rgb layer for each resolution, instead it uses avg_pools to downsample from the highest resolution
        self.bg_pro_upsampler = []
        for i in range(self.progressive_num - 1):
            self.bg_pro_upsampler.append(
                upBlock(init_features // (2 ** i), init_features // (2 ** (i+1)))
            )
        self.bg_pro_upsampler = nn.Sequential(*self.bg_pro_upsampler)
        self.bg_rgb = RGBLayer(init_features // (2 ** (self.progressive_num - 1)))
        self.bg_down_pools = nn.ModuleList(
            nn.AdaptiveAvgPool2d((64 * (2 ** i), 64 * (2 ** i))) for i in range(self.progressive_num - 1)
        )

        # ob_pro_upsamplers apply a rgb layer for each resolution
        self.ob_pro_upsamplers = nn.ModuleList(
            WordLevelCombinedUpsampler(init_features, t_dim) \
                for i in range(self.progressive_num - 1)
        )
        self.ob_rgbs = nn.ModuleList(
            RGBLayer(init_features) for i in range(self.progressive_num)
        )
        self.ob_msks = nn.ModuleList(
            BinaryLayer(init_features) for i in range(self.progressive_num)
        )

    def forward(self, z, t, w, w_mask, bgnoise):
        obs         = []
        ob_masks    = []
        fake_images = []

        s, mu, log_var = self.ca(t)
        
        bg_code = self.bg_fc(torch.cat((z,self.bg_dropout(s)), dim=1)).view(-1, self.base_features, 4, 4)
        ob_code = self.ob_fc(torch.cat((bgnoise,s), dim=1)).view(-1, self.base_features, 4, 4)

        # co-upsample to 64x64
        for co_upsampler in self.init_co_upsamplers:
            bg_code, ob_code = co_upsampler(bg_code, ob_code)
        
        # get the final bg image
        bg_image = self.bg_rgb(self.bg_pro_upsampler(bg_code))

        # progressive upsample
        for i in range(self.progressive_num):
            ob   = self.ob_rgbs[i](ob_code)
            mask = self.ob_msks[i](ob_code)
            if i > self.progressive_num - 2: # no need apply avgpool for final background image
                fake_image = ob * mask + bg_image * (1 - mask)
            else:
                fake_image = ob * mask + self.bg_down_pools[i](bg_image) * (1 - mask)
            fake_images.append(fake_image)
            ob_masks.append(mask)
            obs.append(ob)


            if i < self.progressive_num - 1:
                ob_code = self.ob_pro_upsamplers[i](ob_code, w, w_mask)

        return bg_image, fake_images, ob_masks, mu, log_var, obs

'''
class Generator(nn.Module):
    def __init__(self, z_dim, t_dim, s_dim = 256, resolution = 128, noise_stage_num = 1, base_features = 1024):
        super(Generator, self).__init__()

        assert resolution in (64, 128, 256), 'Resolution should be 64, 128 or 256'
        
        self.noise_stage_num = noise_stage_num
        self.base_features = base_features
        self.progressive_num = int(log(resolution // 64, 2)) + 1
        
        self.ca = ConditionAugmentationModule(t_dim = t_dim, s_dim = s_dim)

        self.bg_fc = nn.Sequential(
            nn.Linear(in_features = z_dim + s_dim, out_features = base_features * 4 * 4 * 2, bias = False),
            nn.BatchNorm1d(base_features * 4 * 4 * 2),
            GLU(),
        ) # latent code -> 4x4

        self.ob_fc = nn.Sequential(
            nn.Linear(in_features = z_dim + s_dim, out_features = base_features * 4 * 4 * 2, bias = False),
            nn.BatchNorm1d(base_features * 4 * 4 * 2),
            GLU()
        ) # latent code -> 4x4

        self.init_co_upsamplers = nn.ModuleList([
            CooperationUpsampleModule(base_features,      base_features),       #   8x8
            CooperationUpsampleModule(base_features // 2, base_features // 2),  # 16x16
            CooperationUpsampleModule(base_features // 4, base_features // 4),  # 32x32
            CooperationUpsampleModule(base_features // 8, base_features // 8)   # 64x64
        ])
        
        init_features = base_features // 16

        # bg_pro_upsampler doesn't apply a rgb layer for each resolution, instead it uses avg_pools to downsample from the highest resolution
        self.bg_pro_upsamplers = []
        for i in range(self.progressive_num - 1):
            self.bg_pro_upsamplers.append(
                WordLevelCombinedUpsampler(init_features, t_dim)
            )
        self.bg_pro_upsamplers = nn.Sequential(*self.bg_pro_upsamplers)
        self.bg_rgb = RGBLayer(init_features)
        self.bg_down_pools = nn.ModuleList(
            nn.AdaptiveAvgPool2d((64 * (2 ** i), 64 * (2 ** i))) for i in range(self.progressive_num - 1)
        )

        # ob_pro_upsamplers apply a rgb layer for each resolution
        self.ob_pro_upsamplers = nn.ModuleList(
            WordLevelCombinedUpsampler(init_features, t_dim) \
                for i in range(self.progressive_num - 1)
        )
        self.ob_rgbs = nn.ModuleList(
            RGBLayer(init_features) for i in range(self.progressive_num)
        )
        self.ob_msks = nn.ModuleList(
            BinaryLayer(init_features) for i in range(self.progressive_num)
        )

    def forward(self, z, t, w, w_mask, bgnoise):
        obs         = []
        ob_masks    = []
        fake_images = []

        s, mu, log_var = self.ca(t)
        
        bg_code = self.bg_fc(torch.cat((z,s), dim=1)).view(-1, self.base_features, 4, 4)
        ob_code = self.ob_fc(torch.cat((z,s), dim=1)).view(-1, self.base_features, 4, 4)

        # co-upsample to 64x64
        for co_upsampler in self.init_co_upsamplers:
            bg_code, ob_code = co_upsampler(bg_code, ob_code)
        
        # get the final bg image
        for i in range(self.progressive_num - 1):
            bg_code = self.bg_pro_upsamplers[i](bg_code, w, w_mask)
        bg_image = self.bg_rgb(bg_code)

        # progressive upsample
        for i in range(self.progressive_num):
            ob   = self.ob_rgbs[i](ob_code)
            mask = self.ob_msks[i](ob_code)
            if i > self.progressive_num - 2: # no need apply avgpool for final background image
                fake_image = ob * mask + bg_image * (1 - mask)
            else:
                fake_image = ob * mask + self.bg_down_pools[i](bg_image) * (1 - mask)
            fake_images.append(fake_image)
            ob_masks.append(mask)
            obs.append(ob)

            if i < self.progressive_num - 1:
                ob_code = self.ob_pro_upsamplers[i](ob_code, w, w_mask)

        return bg_image, fake_images, ob_masks, mu, log_var, obs
'''

# ############## D networks ################################################
def D_Conv3x3Layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, \
            kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def D_Conv7x7Layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, \
            kernel_size = 7, stride = 1, padding = 3, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def D_DownSamplex2Layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, \
            kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.AvgPool2d(kernel_size = 2, stride = 2)
    )

def Block3x3_leakRelu(in_channels, out_channels):
    block = nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class BackgroundDiscriminator(nn.Module):
    def __init__(self, base_features=64):
        super(BackgroundDiscriminator, self).__init__()
        self.image_encoder, self.mask_encoder, self.bf_logits, self.rf_logits = BackgroundDiscriminator.__getEncoders(base_features)

    @staticmethod
    def __getEncoders(base_features):  # Defines the encoder network used for background image
        image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_features, \
                kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=base_features, out_channels=base_features * 2, \
                kernel_size=4, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=base_features * 2, out_channels=base_features * 4, \
                kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        bf_logits = nn.Sequential(
            nn.Conv2d(base_features * 4, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )
        rf_logits = nn.Sequential(
            nn.Conv2d(base_features * 4, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )
        mask_encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=1, padding=0)
        )

        return image_encoder, mask_encoder, bf_logits, rf_logits

    def getPatchLevelLabel(self, mask):
        return self.mask_encoder(mask)

    def forward(self, x):
        code = self.image_encoder(x)
        rf_score = self.rf_logits(code)
        bf_score = self.bf_logits(code)
        return bf_score, rf_score

class ConditionalImageDiscriminator(nn.Module):
    def __init__(self, in_size, text_dim, base_features=64):
        super(ConditionalImageDiscriminator, self).__init__()

        self.text_dim = text_dim
        self.encoder, features = ConditionalImageDiscriminator.__getEncoders(in_size, base_features, text_dim)
        self.jointConv = Block3x3_leakRelu(in_channels = features + text_dim, out_channels = features)
        self.logits = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=1, kernel_size=4, stride=4, padding=0, bias=True),
            nn.Sigmoid()
        )

    @staticmethod
    def __getEncoders(in_size, base_features, text_dim):
        encoder = [
            D_Conv7x7Layer(in_channels = 3, out_channels = base_features),
            D_Conv3x3Layer(in_channels = base_features, out_channels = base_features)
        ] # we find quick down-sample may destroy image information, so we applied convolution ops first

        while in_size > 4:
            encoder.append(D_DownSamplex2Layer(in_channels = base_features, out_channels = base_features * 2)),
            base_features *= 2
            in_size = in_size // 2

        assert in_size == 4
        return nn.Sequential(*encoder), base_features

    def forward(self, x, t):
        x_code = self.encoder(x)
        t_code = t.view(-1, self.text_dim, 1, 1).repeat(1, 1, 4, 4)
        xt_code = self.jointConv(torch.cat((x_code, t_code), 1))
        return self.logits(xt_code)

#######################################################################
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        self.rnn = nn.LSTM(self.ninput, self.nhidden,
                           self.nlayers, batch_first=True,
                           dropout=self.drop_prob,
                           bidirectional=self.bidirectional)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()))

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        #emb = torch.reshape(emb, (1, *emb.shape))
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb

class CNN_ENCODER(nn.Module):
    def __init__(self, nef = 256):
        super(CNN_ENCODER, self).__init__()
        self.nef = nef
        model = models.inception_v3(init_weights=False)
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code

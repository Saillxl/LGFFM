import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from LGFFM.common import FDMM, Local_block, MMF


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LGFFM(nn.Module):
    def __init__(self, checkpoint_path=None,decode_channels=96,dropout=0.1,window_size=8,num_classes=1,
                 conv_depths=(2, 2, 2, 2), conv_dims=(144, 288, 576, 1152), conv_drop_path_rate=0.) -> None:
        super(LGFFM, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.rfb1 = RFB_modified(144, 96)
        self.rfb2 = RFB_modified(288, 96)
        self.rfb3 = RFB_modified(576, 96)
        self.rfb4 = RFB_modified(1152, 96)
        self.rfb5 = RFB_modified(144, 96)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               nn.Conv2d(decode_channels, num_classes, kernel_size=1))
        self.down = nn.Conv2d(in_channels=2*decode_channels,out_channels=decode_channels,kernel_size=1)

        ###### Frequency Domain Mapping Module #######
        self.fuseFeature = FDMM(in_ch=3 * decode_channels, out_ch=decode_channels)


        ###### Local Feature Block #######

        self.downsample_layers = nn.ModuleList()  # stem + 3 stage downsample
        stem = nn.Sequential(nn.Conv2d(3, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # stage2-4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(conv_dims[i], conv_dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(conv_depths[i])]
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        self.align1 = MMF()
        self.align2 = MMF()


    def forward(self, x):
        h, w = x.size()[-2:]
        x1, x2, x3, x4 = self.encoder(x)
        # res1,res2,res3,res4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        
        ######  Local Branch ######
        x_l = self.downsample_layers[0](x)
        x_l_1 = self.stages[0](x_l) + x1
        x_l = self.downsample_layers[1](x_l_1)
        x_l_2 = self.stages[1](x_l) + x2
        x_l = self.downsample_layers[2](x_l_2)
        x_l_3 = self.stages[2](x_l) + x3
        x_l = self.downsample_layers[3](x_l_3)
        x_l_4 = self.stages[3](x_l) + x4

        change_channel = self.rfb5(x1)

        res1, res2, res3, res4 = self.rfb1(x_l_1), self.rfb2(x_l_2), self.rfb3(x_l_3), self.rfb4(x_l_4)
        # res1, res2, res3, res4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        res1h, res1w = res1.size()[-2:]

        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        # res5 = F.interpolate(change_channel, size=(res1h, res1w), mode='bicubic', align_corners=False)

        middleres = torch.cat([res2, res3, res4], dim=1)

        fusefeature_L, fusefeature_H = self.fuseFeature(middleres)
        fusefeature_L = F.interpolate(fusefeature_L, size=(res1h, res1w), mode='bicubic', align_corners=False)
        fusefeature_H = F.interpolate(fusefeature_H, size=(res1h, res1w), mode='bicubic', align_corners=False)

        fusefeature_L = self.align1(fusefeature_L, change_channel)
        fusefeature_H = self.align2(fusefeature_H, res1)

        res = torch.cat([fusefeature_H, fusefeature_L],dim=1)
        # res = torch.cat([res1, res2, res3, res4],dim=1)

        middleres = self.down(res)

        res = self.segmentation_head(middleres)

        x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)

        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = LGFFM(num_classes=2).cuda()
        x = torch.randn(1, 3, 256, 256).cuda()
        out = model(x)
        print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from timm.models.layers import DropPath



class Local_block(nn.Module):
    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + self.drop_path(x)
        return x



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



class FDMM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FDMM, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # self.glb = GlBlock()
        #
        # self.localb = LoBlock()

        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # self.outconv_bn_relu_glb = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        # )
        # self.outconv_bn_relu_local = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        # )
    def forward(self, x):

        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        yH = y_HL + y_LH + y_HH

        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)

        return yL,yH #,glb,local


class MMF(nn.Module):
    def __init__(self, in_ch=96):
        super(MMF, self).__init__()

        self.conv_bn_relu11 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu12 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu21 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu22 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, x, y):
        x1 = self.conv_bn_relu11(x)
        y1 = self.conv_bn_relu12(y)
        b, c, h, w = x.shape
        x1 = x1.reshape(b, c, h * w)
        y1 = y1.reshape(b, c, h * w)
        similarity = torch.bmm(x1.permute(0, 2, 1), y1)
        similarity_softmax = torch.nn.functional.softmax(similarity, dim=2)

        output_x1 = torch.bmm(x1, similarity_softmax)
        output_y1 = torch.bmm(y1, similarity_softmax)

        similarity_sum = similarity.sum(dim=2)
        similarity_sums_sig = torch.sigmoid(similarity_sum)
        similarity_sums_sig_expanded = similarity_sums_sig.unsqueeze(1)
        output_x1 = output_x1 * (1-similarity_sums_sig_expanded)
        output_y1 = output_y1 * (1-similarity_sums_sig_expanded)

        output_x1 = output_x1.reshape(b, c, h, w)
        output_y1 = output_y1.reshape(b, c, h, w)
        output_x1 = self.conv_bn_relu21(output_x1)
        output_y1 = self.conv_bn_relu22(output_y1)
        output = output_x1 + output_y1 + x + y

        return output

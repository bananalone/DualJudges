from net import DeepLabv3_plus, SegNet, UNet

from factory import register_model


@register_model
def deeplabv3_plus(in_channels, out_channels, os=16, pretrained=False, freeze_bn=False, _print=True):
    return DeepLabv3_plus(in_channels, out_channels, os, pretrained, freeze_bn, _print)


@register_model
def segnet(in_channels,out_channels):
    return SegNet(in_channels,out_channels)


@register_model
def unet(in_channels, out_channels):
    return UNet(in_channels, out_channels)
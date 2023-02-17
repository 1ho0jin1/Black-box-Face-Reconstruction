import os
import yaml
import torch
from encoder.VGGNets import VGG
from encoder.ResNets import Resnet
from encoder.Swin_Transformer import SwinTransformer



def fetch_encoder(encoder_type, pretrained=True,
                  encoder_conf_file=f"{os.path.dirname(__file__)}/encoder_conf.yaml"):
    with open(encoder_conf_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf = conf[encoder_type]
    print('encoder param:', conf)

    if encoder_type == 'VGGNet19':
        encoder = VGG('VGG19')

    elif encoder_type == 'ResNet50':
        depth = conf['depth']  # depth of the ResNet, e.g. 50, 100, 152.
        drop_ratio = conf['drop_ratio']  # drop out ratio.
        net_mode = conf['net_mode']  # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
        feat_dim = conf['feat_dim']  # dimension of the output features, e.g. 512.
        out_h = conf['out_h']  # height of the feature map before the final features.
        out_w = conf['out_w']  # width of the feature map before the final features.
        encoder = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)

    elif encoder_type == 'SwinTransformer':
        img_size = conf['img_size']
        patch_size = conf['patch_size']
        in_chans = conf['in_chans']
        embed_dim = conf['embed_dim']
        depths = conf['depths']
        num_heads = conf['num_heads']
        window_size = conf['window_size']
        mlp_ratio = conf['mlp_ratio']
        drop_rate = conf['drop_rate']
        drop_path_rate = conf['drop_path_rate']
        encoder = SwinTransformer(img_size=img_size,
                                   patch_size=patch_size,
                                   in_chans=in_chans,
                                   embed_dim=embed_dim,
                                   depths=depths,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=drop_rate,
                                   drop_path_rate=drop_path_rate,
                                   ape=False,
                                   patch_norm=True,
                                   use_checkpoint=False)
    elif encoder_type == 'FaceNet':
        from facenet_pytorch import InceptionResnetV1
        encoder = InceptionResnetV1(pretrained='vggface2')
    else:
        raise NotImplementedError(f"{encoder_type} is not implemented!")

    # save image size & align info.
    encoder.align = conf['align']
    encoder.img_size = conf['img_size']

    # activate eval mode
    encoder.eval()

    if pretrained and encoder_type not in ['FaceNet', 'HOG']:
        stdict = torch.load(conf['weight'], map_location='cpu')
        encoder.load_state_dict(stdict)

    return encoder
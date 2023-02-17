import torch
from generator.stylegan2 import Generator


stylegan_conf = {
    'FFHQ-256':{
        'resolution': 256,
        'weight': '/data/New_Projects/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-256.pt',
    }
}


def fetch_stylegan(generator_type):
    conf = stylegan_conf[generator_type]
    resolution = conf['resolution']
    weight = conf['weight']
    stdict = torch.load(weight, map_location='cpu')

    # default parameters (https://github.com/rosinality/stylegan2-pytorch)
    latent = 512
    n_mlp = 8
    channel_multiplier = 2

    generator = Generator(resolution, latent, n_mlp, channel_multiplier=channel_multiplier)
    generator.load_state_dict(stdict["g_ema"], strict=False)
    generator.eval()

    return generator
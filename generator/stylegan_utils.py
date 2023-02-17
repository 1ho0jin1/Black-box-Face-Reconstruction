import torch
import torch.nn as nn
from generator.stylegan2 import Generator


stylegan_conf = {
    'FFHQ-256':{
        'resolution': 256,
        'weight': '/data/New_Projects/stylegan2-pytorch/checkpoint/stylegan2-ffhq-config-256.pt',
    }
}



class StyleGANWrapper(nn.Module):
    """
    Wrapper class for StyleGAN.
    Performs processing & cropping in the forward function.
    """
    def __init__(self, args):
        super(StyleGANWrapper, self).__init__()
        self.generator_type = args.generator_type
        self.truncation = args.truncation
        self.crop_size = args.crop_size
        self.device = args.device

        # fetch and initialize StyleGAN
        self.fetch_stylegan()
        self.get_avg_latent()


    def fetch_stylegan(self):
        conf = stylegan_conf[self.generator_type]
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
        self.generator = generator.to(self.device)


    @ torch.no_grad()
    def get_avg_latent(self):
        avg_latent = self.generator.mean_latent(int(1e4))
        self.avg_latent = avg_latent.unsqueeze(1).repeat(1, self.generator.n_latent, 1)


    def postprocess(self, img, v_offset=10):
        '''
        postprocessing for StyleGAN-FFHQ-256
        crops and normalizes the generated image
        '''
        _, _, cy, cx = img.shape
        assert len(img.shape) == 4 and img.shape[1] == 3, 'img must be a Bx3xHxW numpy array'
        assert cy >= self.crop_size and cx >= self.crop_size, 'crop size must be smaller than the given image'
        cy = cy // 2 + v_offset  # vertical offset
        cx = cx // 2
        w = self.crop_size // 2
        img = img[:, :, cy - w:cy + w, cx - w:cx + w]
        img = 2 * (img - img.min()) / (img.max() - img.min()) - 1  # normalize -1~1
        return img


    def forward(self, latent, truncation_latent=None, return_latents=False):
        truncation = 1.0 if truncation_latent is None else self.truncation
        img, ltn = self.generator([latent],
                                  input_is_latent=True,
                                  randomize_noise=False,
                                  truncation=truncation,
                                  truncation_latent=truncation_latent,
                                  return_latents=True)
        img = self.postprocess(img)
        if return_latents:
            return img, ltn
        else:
            return img

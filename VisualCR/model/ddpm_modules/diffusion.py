import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(
            'Only "linear" and "cosine" beta schedules are supported, got: {}'.format(schedule)
        )
    return betas

def extract(a, t, x_shape):
    bs, = t.shape
    assert x_shape[0] == bs
    if not torch.is_tensor(a):
        a = torch.as_tensor(a, dtype=torch.float32, device=t.device)
    elif a.device != t.device:
        a = a.to(t.device)
    out = a.gather(0, t.long())
    assert out.shape == (bs,)
    return out.reshape((bs,) + (1,) * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        ddim_steps=5,
        lambda_feat=0.01
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.lambda_feat = float(lambda_feat)
        self.loss_terms = {}
        self.default_ddim_steps = max(1, int(ddim_steps))
        object.__setattr__(self, '_feature_extractor', None)
        self._build_feature_extractor()

    def _build_feature_extractor(self):
        try:
            from model.ddpm_modules import unet as unet_module
        except Exception as exc:
            raise RuntimeError(
                'Perceptual feature extractor is unavailable: failed to import '
                '`model/ddpm_modules/unet.py`.'
            ) from exc

        feature_cls = getattr(unet_module, 'Discriminator', None)
        feature_cls_name = 'Discriminator'
        if feature_cls is None:
            feature_cls = getattr(unet_module, 'VGGFeatureExtractor', None)
            feature_cls_name = 'VGGFeatureExtractor'
        if feature_cls is None:
            raise RuntimeError(
                'Perceptual feature extractor is unavailable: expected '
                '`Discriminator` or `VGGFeatureExtractor` in `model/ddpm_modules/unet.py`.'
            )

        try:
            if feature_cls_name == 'VGGFeatureExtractor':
                feature_extractor = feature_cls(use_enhance=False)
            else:
                feature_extractor = feature_cls()
        except Exception as exc:
            raise RuntimeError(
                'Perceptual feature extractor is unavailable: failed to initialize '
                '`{}` (check its dependencies/weights availability).'.format(feature_cls_name)
            ) from exc

        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()

        # Keep this module out of the diffusion state_dict so older checkpoints
        # remain loadable without strict=False.
        object.__setattr__(self, '_feature_extractor', feature_extractor)

    def _prepare_feature_input(self, x):
        x_detached = x.detach()
        if x_detached.amin().item() < 0.0 or x_detached.amax().item() > 1.0:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
        else:
            raise NotImplementedError()

        if self._feature_extractor is None:
            raise RuntimeError(
                'Perceptual feature extractor is unavailable in GaussianDiffusion. '
                'Cannot compute feature loss.'
            )
        feature_extractor = self._feature_extractor.to(device)
        feature_extractor.eval()
        object.__setattr__(self, '_feature_extractor', feature_extractor)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))

    def _require_condition_inputs(self, condition_x, template, context):
        if condition_x is None:
            raise ValueError('Missing required conditioning tensor "SR" for {}.'.format(context))
        if template is None:
            raise ValueError('Missing required template tensor "TEMPLATE" for {}.'.format(context))

    def p_sample_ddim(self, x, t, t_next, condition_x, template):
        at = extract(self.alphas_cumprod, t, x.shape)
        self._require_condition_inputs(condition_x, template, 'diffusion sampling')
        et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, template)

        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        if t_next is None:
            at_next = torch.ones_like(at)
        else:
            at_next = extract(self.alphas_cumprod, t_next, x.shape)
        return at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et

    def _build_default_time_steps(self, num_steps):
        num_steps = max(1, min(int(num_steps), self.num_timesteps))
        time_steps = np.linspace(0, self.num_timesteps - 1, num_steps, dtype=np.int64)
        time_steps = np.unique(time_steps)
        return np.flip(time_steps)

    @torch.no_grad()
    def p_sample_loop(self, x_in):
        device = self.betas.device
        g_gpu = torch.Generator(device=device).manual_seed(44444)
        condition_x = x_in.get('SR')
        template = x_in.get('TEMPLATE')
        self._require_condition_inputs(condition_x, template, 'diffusion sampling')
        b = condition_x.shape[0]
        img = torch.randn(condition_x.shape, device=device, generator=g_gpu)

        time_steps = self._build_default_time_steps(self.default_ddim_steps)
        for j, i in enumerate(time_steps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if j == len(time_steps) - 1:
                t_next = None
            else:
                t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
            img = self.p_sample_ddim(img, t, t_next, condition_x=condition_x, template=template)
        return img

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )


    def p_losses(self, x_in, noise=None):
        x_start = x_in.get('HR')
        condition_x = x_in.get('SR')
        template = x_in.get('TEMPLATE')
        if x_start is None:
            raise ValueError('Missing required target tensor "HR" for diffusion training.')
        self._require_condition_inputs(condition_x, template, 'diffusion training')
        b = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(
            torch.cat([condition_x, x_noisy], dim=1), t, template)

        noise_loss = self.loss_func(noise, x_recon)
        alpha_t = extract(self.alphas_cumprod, t, x_noisy.shape)
        x0_pred = (
            x_noisy - torch.sqrt(1.0 - alpha_t) * x_recon
        ) / torch.sqrt(alpha_t)

        pred_feat_input = self._prepare_feature_input(x0_pred)
        pred_features = self._feature_extractor(pred_feat_input)
        with torch.no_grad():
            target_features = self._feature_extractor(self._prepare_feature_input(x_start))

        if not isinstance(pred_features, (list, tuple)) or not isinstance(target_features, (list, tuple)):
            raise RuntimeError(
                'Perceptual feature extractor must return a list/tuple of feature maps.'
            )
        if len(pred_features) != len(target_features):
            raise RuntimeError(
                'Perceptual feature extractor returned mismatched feature pyramid lengths '
                'between prediction and target branches.'
            )

        feat_loss = x0_pred.new_tensor(0.0)
        for pred_feature, target_feature in zip(pred_features, target_features):
            feat_loss = feat_loss + F.l1_loss(pred_feature, target_feature)

        loss = noise_loss + self.lambda_feat * feat_loss

        self.loss_terms = {
            'l_noise': noise_loss.detach(),
            'l_feat': feat_loss.detach(),
            'l_total': loss.detach()
        }
        return loss

    def get_loss_terms(self):
        return self.loss_terms

    def forward(self, x):
        return self.p_losses(x)

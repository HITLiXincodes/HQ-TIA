"""Canonical model stack implementation for this package."""

import logging
from collections import OrderedDict
import os
import torch

logger = logging.getLogger('base')


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for idx, item in enumerate(x):
                if item is not None:
                    x[idx] = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n


def define_G(opt):
    model_opt = opt['model']
    from .ddpm_modules import diffusion, unet
    model = unet.UNet(
        template_dim=model_opt['diffusion'].get('template_dim', 512)
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',
        ddim_steps=model_opt['diffusion'].get('ddim_steps', 20),
        lambda_feat=model_opt['diffusion'].get('lambda_feat', 0.01)
    )
    return netG


class DDPM(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.netG = self.set_device(define_G(opt))
        self.schedule_phase = None

        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()

            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        if data.get('SR') is None:
            raise ValueError('Missing required conditioning tensor "SR" in model input batch.')
        if data.get('TEMPLATE') is None:
            raise ValueError('Missing required template tensor "TEMPLATE" in model input batch.')
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        l_pix.backward()
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()
        loss_terms = self.netG.get_loss_terms()
        if loss_terms:
            if 'l_noise' in loss_terms:
                self.log_dict['l_noise'] = loss_terms['l_noise'].item()
            if 'l_feat' in loss_terms:
                self.log_dict['l_feat'] = loss_terms['l_feat'].item()
            if 'l_total' in loss_terms:
                self.log_dict['l_total'] = loss_terms['l_total'].item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.SR = self.netG.p_sample_loop(self.data)

        self.netG.train()

    def set_loss(self):
        self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach().float().cpu()
        out_dict['INF'] = self.data['SR'].detach().float().cpu()
        out_dict['HR'] = self.data['HR'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        state_dict = self.netG.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            self.netG.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']


def create_model(opt):
    m = DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

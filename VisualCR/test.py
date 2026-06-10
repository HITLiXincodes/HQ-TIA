import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core.metrics import tensor2img, save_img
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config/VisualCR.json'),
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='Batch size for validation inference')
    parser.add_argument('-debug', '-d', action='store_true')

    args = parser.parse_args()
    args.phase = 'val'
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'inference', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    val_dataset_opt = opt['datasets'].get('val')
    if val_dataset_opt is None:
        raise ValueError('Missing required "val" dataset configuration.')
    if args.batch_size is not None:
        val_dataset_opt['batch_size'] = args.batch_size

    val_input_root = val_dataset_opt['input_root']
    val_set = Data.create_dataset(val_dataset_opt)
    val_loader = Data.create_dataloader(val_set, val_dataset_opt, 'val')
    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    ddim_steps = int(opt['model']['diffusion'].get('ddim_steps', diffusion.netG.default_ddim_steps))
    diffusion.netG.default_ddim_steps = max(1, ddim_steps)
    logger.info('Initial Model Finished')
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    logger.info('Begin Model Inference | ddim_steps={}'.format(diffusion.netG.default_ddim_steps))

    result_path = opt['path']['results']
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        diffusion.feed_data(val_data)
        diffusion.test()
        visuals = diffusion.get_current_visuals()

        sr_batch = visuals['SR']
        indices = val_data['Index'].tolist()

        for b_idx, sample_index in enumerate(indices):
            sample_index = int(sample_index)
            input_path = val_loader.dataset.input_path[sample_index]
            save_rel_path = os.path.splitext(os.path.relpath(input_path, val_input_root))[0] + '.jpg'
            save_path = os.path.join(result_path, save_rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            sr_img = tensor2img(sr_batch[b_idx])
            save_img(sr_img, save_path)

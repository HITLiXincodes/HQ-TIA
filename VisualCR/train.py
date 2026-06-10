import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os


def _ensure_batched(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def run_periodic_validation(diffusion, val_loader, opt, logger, current_epoch, current_step):
    result_tag = 'train_epoch{:04d}_iter{:08d}'.format(current_epoch, current_step)
    result_path = os.path.join(opt['path']['results'], result_tag)
    os.makedirs(result_path, exist_ok=True)

    logger.info(
        '[validation] start | epoch={} iter={} | inference=test() | save_dir={}'.format(
            current_epoch, current_step, result_path))

    sample_count = 0
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    for _, val_data in enumerate(val_loader):
        diffusion.feed_data(val_data)
        diffusion.test()
        visuals = diffusion.get_current_visuals()

        sr_batch = _ensure_batched(visuals['SR'])
        hr_batch = _ensure_batched(visuals['HR'])
        inf_batch = _ensure_batched(visuals['INF'])

        index_batch = val_data.get('Index', None)
        if index_batch is not None and hasattr(index_batch, 'dim') and index_batch.dim() == 0:
            index_batch = index_batch.unsqueeze(0)

        batch_size = sr_batch.shape[0]
        for batch_idx in range(batch_size):
            sample_count += 1
            sample_idx = sample_count
            if index_batch is not None:
                sample_idx = int(index_batch[batch_idx])
            sample_prefix = 'iter{:08d}_sample{:06d}_idx{:06d}'.format(
                current_step, sample_count, sample_idx)

            restore_img = Metrics.tensor2img(sr_batch[batch_idx])
            target_img = Metrics.tensor2img(hr_batch[batch_idx])
            input_img = Metrics.tensor2img(inf_batch[batch_idx])

            Metrics.save_img(
                target_img, os.path.join(result_path, '{}_target.png'.format(sample_prefix)))
            Metrics.save_img(
                restore_img, os.path.join(result_path, '{}_restore.png'.format(sample_prefix)))
            Metrics.save_img(
                input_img, os.path.join(result_path, '{}_input.png'.format(sample_prefix)))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')

    logger.info(
        '[validation] end | epoch={} iter={} | processed_samples={} | save_dir={}'.format(
            current_epoch, current_step, sample_count, result_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config/VisualCR.json'),
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    args.phase = 'train'
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    train_dataset_opt = opt['datasets'].get('train')
    val_dataset_opt = opt['datasets'].get('val')
    if train_dataset_opt is None:
        raise ValueError('Missing required "train" dataset configuration.')
    if val_dataset_opt is None:
        raise ValueError('Missing required "val" dataset configuration for periodic validation.')

    train_set = Data.create_dataset(train_dataset_opt)
    train_loader = Data.create_dataloader(train_set, train_dataset_opt, 'train')
    val_set = Data.create_dataset(val_dataset_opt)
    val_loader = Data.create_dataloader(val_set, val_dataset_opt, 'val')
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')
    logger.info('Run mode: train')
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()

            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                logger.info(
                    '[train] epoch={} iter={} | losses={}'.format(
                        current_epoch, current_step, logs))

            if current_step % opt['train']['val_freq'] == 0:
                run_periodic_validation(
                    diffusion=diffusion,
                    val_loader=val_loader,
                    opt=opt,
                    logger=logger,
                    current_epoch=current_epoch,
                    current_step=current_step
                )

            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

    logger.info('End of training.')

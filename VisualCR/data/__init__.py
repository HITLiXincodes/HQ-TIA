'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt.get('batch_size', 1),
            shuffle=False,
            num_workers=dataset_opt.get('num_workers', 1),
            pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt):
    '''create dataset'''
    from data.dataset import Dataset as D
    dataset = D(input_root=dataset_opt.get('input_root'),
                target_root=dataset_opt.get('target_root'),
                template_root=dataset_opt.get('template_root'),
                data_len=dataset_opt['data_len']
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

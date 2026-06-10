from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import data.util as Util


class Dataset(TorchDataset):
    def __init__(self, input_root, target_root, template_root, data_len=-1):
        self.input_path = Util.get_paths_from_images(input_root)
        self.target_path = Util.get_paths_from_images(target_root)
        self.template_path = self._build_template_paths(
            input_root=input_root,
            template_root=template_root,
            input_paths=self.input_path
        )

        self.dataset_len = len(self.target_path)
        if data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def _build_template_paths(self, input_root, template_root, input_paths):
        if template_root is None:
            raise ValueError('Missing template_root. Expected a directory containing per-image .npy templates.')
        return [
            os.path.join(
                template_root,
                os.path.splitext(os.path.relpath(input_path, input_root))[0] + '.npy'
            )
            for input_path in input_paths
        ]

    def _load_template(self, template_path):
        if not template_path.endswith('.npy'):
            raise ValueError(
                'Invalid template path "{}". Expected a .npy template file.'.format(template_path)
            )
        if not os.path.isfile(template_path):
            raise FileNotFoundError(
                'Template file not found: "{}". Expected a .npy template file.'.format(template_path)
            )
        try:
            template_array = np.load(template_path, allow_pickle=False)
            template_tensor = torch.from_numpy(np.asarray(template_array)).float().view(-1)
        except Exception as exc:
            raise ValueError(
                'Failed to load template file "{}". Expected a valid .npy template array.'.format(template_path)
            ) from exc
        return template_tensor

    def __getitem__(self, index):
        target = Image.open(self.target_path[index]).convert("RGB")
        input_img = Image.open(self.input_path[index]).convert("RGB")
        template = self._load_template(self.template_path[index])
        target = Util.image_to_tensor(target, min_max=(-1, 1))
        input_img = Util.image_to_tensor(input_img, min_max=(-1, 1))
        return {'HR': target, 'SR': input_img, 'TEMPLATE': template, 'Index': index}

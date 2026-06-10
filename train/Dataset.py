# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
'''
Note: If you use this implementation, please cite the following paper:
- Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, and Sébastien Marcel. "Vulnerability of
  State-of-the-Art Face Recognition Models to Template Inversion Attack", IEEE Transactions 
  on Information Forensics and Security, 2024.
'''
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os 
from torchvision import transforms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):
    def __init__(self, dataset_dir = None,
                       train=True,
                       device='cpu',
                       train_test_split = 0.9,
                ):
        if dataset_dir is None:
            raise ValueError("dataset_dir must be provided. Set paths.dataset_dir in config or pass it explicitly.")
        self.dataset_dir = dataset_dir
        self.device = device
        self.train  = train

        self.dir_all_images = []
        all_npyfiles = glob.glob(dataset_dir+'/images/*')
        all_npyfiles.sort()
        for npyfile in all_npyfiles:
            self.dir_all_images.append(os.path.basename(npyfile))
            
        if self.train:
            self.dir_all_images = self.dir_all_images[:int(train_test_split*len(self.dir_all_images))]
        else:
            self.dir_all_images = self.dir_all_images[int(train_test_split*len(self.dir_all_images)):]


    def __len__(self):
        return len(self.dir_all_images)

    def __getitem__(self, idx):

        image = f"{self.dataset_dir}/images/{self.dir_all_images[idx]}"
        embedding = f"{self.dataset_dir}/embeddings/{self.dir_all_images[idx]}"

        image     = np.load(image)
        embedding = np.load(embedding)
        
        image     = self.transform_image(image)
        embedding = self.transform_embedding(embedding)
        
        return embedding, image
    
    def transform_image(self,image):
        transform = transforms.ToTensor()
        image = transform(image).to(self.device)
        image = image*2 -1
        return image
    
    def transform_embedding(self, embedding):
        embedding = torch.Tensor(embedding).to(self.device)
        return embedding

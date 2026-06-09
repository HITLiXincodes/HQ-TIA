import argparse
import torch

parser = argparse.ArgumentParser(
    description='Vulnerability evaluation of face reocgnition system against template inversion attack')
parser.add_argument('--FR_system', metavar='<FR_system>', type=str, default='adaface',
                    help='adaface/ElasticFace')
device = 'cuda:3'
# ================== load FR system ======================
# ================== load reconstruction model ======================
# ================== name as inv_transformer ======================
class InversionTransformer():
    def __init__(self, checkpoint, network):

        self.device = device
        self.generator = network()
        self.generator.load_state_dict(
            torch.load(checkpoint, map_location=self.device,)
        )
        self.generator.eval()
        self.generator.to(self.device)

    def transform(self, data):
        data = np.reshape(data, (1, data.shape[0], 1, 1))
        embedding = torch.Tensor(data).to(self.device)
        reconstructed_img = self.generator(embedding)[0]
        #reconstructed_img = (reconstructed_img+1)/2
        return reconstructed_img.cpu().detach().numpy() * 255.0

import os
epoch = 90
generator_checkpoint = f'/home/ubuntu/FR_Attack/pixel_stage1/adaface/training_files/models/Generator_91.pth'

import os,sys
sys.path.append('/home/ubuntu/FR_Attack/pixel_stage1/adaface')
from src.Network import Generator
inv_transformer = InversionTransformer(checkpoint=generator_checkpoint, network=Generator)
# ================== dataset ======================
# ================== get data names and folders ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

testing_dataset = MyDataset(train=True, device=device, dataset_dir=f'/home/ubuntu/FR_Attack/databases/AgeDB/adaface_112',train_test_split = 1.0)
test_dataloader = DataLoader(testing_dataset, batch_size=128, shuffle=False)
# ================== storage_real_embeddings and images ======================
import cv2
import torch
import numpy as np
import os, sys
sys.path.append("/home/ubuntu/FR_Attack/data_gen")
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
count1=0
count2=0
for embedding, real_image in test_dataloader:
    real_image = real_image.cpu()
    for i in range(real_image.size(0)):
        os.makedirs(f'real_agedb/', exist_ok=True)
        img = real_image[i].squeeze()
        img = (img+1)/2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(f'real_agedb/{count1}.jpg',
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        count1+=1
    fake_image = inv_transformer.generator(embedding).detach().cpu()
    for i in range(fake_image.size(0)):
        os.makedirs(f'fake_agedb/', exist_ok=True)
        img = fake_image[i].squeeze()
        img = (img+1)/2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(f'fake_agedb/{count2}.jpg',
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        count2+=1

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train.Dataset import MyDataset
from train.Network import Generator
from train.config import add_config_arguments, load_config, resolve_device

parser = argparse.ArgumentParser(
    description='Vulnerability evaluation of face reocgnition system against template inversion attack')

add_config_arguments(parser)
args = parser.parse_args()
config = load_config(args.config, args.set)

device = resolve_device(config["image_generation"]["device"])


def load_generator(checkpoint):
    generator = Generator()
    generator.load_state_dict(torch.load(checkpoint, map_location=device))
    generator.eval()
    generator.to(device)
    return generator


generator_checkpoint = config["image_generation"]["checkpoint"]
generator = load_generator(generator_checkpoint)

testing_dataset = MyDataset(
    train=True,
    device=device,
    dataset_dir=config["image_generation"]["dataset_dir"],
    train_test_split=config["image_generation"]["train_test_split"],
)
test_dataloader = DataLoader(
    testing_dataset,
    batch_size=config["image_generation"]["batch_size"],
    shuffle=False,
    num_workers=config["image_generation"]["num_workers"],
)
count1=0
count2=0
real_output_dir = config["image_generation"]["real_output_dir"]
fake_output_dir = config["image_generation"]["fake_output_dir"]
for embedding, real_image in test_dataloader:
    real_image = real_image.cpu()
    for i in range(real_image.size(0)):
        os.makedirs(real_output_dir, exist_ok=True)
        img = real_image[i].squeeze()
        img = (img+1)/2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(os.path.join(real_output_dir, f"{count1}.jpg"),
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        count1+=1
    fake_image = generator(embedding).detach().cpu()
    for i in range(fake_image.size(0)):
        os.makedirs(fake_output_dir, exist_ok=True)
        img = fake_image[i].squeeze()
        img = (img+1)/2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(os.path.join(fake_output_dir, f"{count2}.jpg"),
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        count2+=1

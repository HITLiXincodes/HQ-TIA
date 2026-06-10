import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(
    description='Vulnerability evaluation of face reocgnition system against template inversion attack')
from train.config import add_config_arguments, load_config, require_config_path, resolve_device

add_config_arguments(parser)
args = parser.parse_args()
config = load_config(args.config, args.set)

device = resolve_device(config["face_system"]["device"])
sys.path.append(require_config_path(config, "paths.face_system_dir"))
from adaface import net

adaface_models = {
    config["face_system"]["adaface_architecture"]: require_config_path(config, "face_system.adaface_checkpoint"),
}

def load_pretrained_model(architecture=None):
    if architecture is None:
        architecture = config["face_system"]["adaface_architecture"]
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], weights_only= False, map_location=device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.to(torch.device(device))
    model.eval()
    return model

face_transformer = load_pretrained_model()

input_root = config["embedding_generation"]["input_root"]
target_root = config["embedding_generation"]["target_root"]
def get_all_images(dataset_dir):
    image_paths = []
    img_files = sorted(os.listdir(dataset_dir))
    for img_file in img_files:
        file_path = os.path.join(dataset_dir, img_file)
        if os.path.isfile(file_path):
            image_paths.append(file_path)
    return image_paths

input_paths = get_all_images(input_root)
target_paths = get_all_images(target_root)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

def transform_embedding(embedding):
    embedding = embedding.cpu().detach().numpy()
    embedding = np.reshape(embedding, [embedding.shape[-1], 1, 1])
    return embedding

def getitems(img_path):
    image = Image.open(img_path).convert("RGB").resize((112, 112))
    transformed_input = transform(image).unsqueeze(0)

    embedding,_ = face_transformer(transformed_input.to(device))

    embedding = transform_embedding(embedding)

    return embedding

def get_features(test_list):
    embeddings = []

    for img_path in test_list:
        embedding = getitems(img_path)
        embeddings.append(embedding)

    return embeddings

def storage_embeddings(embeddings, store_root):
    if not os.path.exists(store_root):
        os.makedirs(store_root)
    for idx in range(len(embeddings)):
        np.save(f'{store_root}/{idx}', embeddings[idx])

input_embeddings = get_features(input_paths)
target_embeddings = get_features(target_paths)

storage_embeddings(input_embeddings, config["embedding_generation"]["fake_embedding_output_dir"])
storage_embeddings(target_embeddings, config["embedding_generation"]["real_embedding_output_dir"])

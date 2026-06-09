import argparse
import torch
import os
parser = argparse.ArgumentParser(
    description='Vulnerability evaluation of face reocgnition system against template inversion attack')
parser.add_argument('--FR_system', metavar='<FR_system>', type=str, default='adaface',
                    help='adaface/ElasticFace')
device = 'cuda:3'
# ================== load FR system ======================
# ================== name as face_transformer ======================
import sys
sys.path.append('/home/ubuntu/FR_Attack/face_system')
from adaface import net

adaface_models = {
    'ir_50':'/home/ubuntu/FR_Attack/face_system/adaface/adaface_ir50_ms1mv2.ckpt',
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], weights_only= False, map_location=device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.to(torch.device(device))
    model.eval()
    return model

face_transformer = load_pretrained_model()

# ================== dataset ======================
# ================== get data names and folders ======================
input_root = '/home/ubuntu/FR_Attack/pixel_stage1/adaface/test/agedb/fake_AgeDB'
target_root = '/home/ubuntu/FR_Attack/pixel_stage1/adaface/test/agedb/real_AgeDB'
def get_all_images(dataset_dir):
    files = []
    full_name = []
    img_files = sorted(os.listdir(dataset_dir))
    for img_file in img_files:
        file_path = os.path.join(dataset_dir, img_file)
        if os.path.isfile(file_path):
            files.append(img_file)  # 原来的 class_name 设置为 None
            full_name.append(file_path)
    return full_name, files

input_paths, input_folders = get_all_images(input_root)
target_paths, target_folders = get_all_images(target_root)
#print(img_paths,image_folders)
# ================== storage_real_embeddings and images ======================
import cv2
import torch
import numpy as np
import os, sys
sys.path.append("/home/ubuntu/FR_Attack/datagen")
from face_alignment import align
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

def transform_image(image):
    return image

def transform_embedding(embedding):
    embedding = embedding.cpu().detach().numpy()
    embedding = np.reshape(embedding, [embedding.shape[-1], 1, 1])
    return embedding

def getitems(img_path):
    aligned = align.get_aligned_face(img_path)
    transformed_input = transform(aligned).unsqueeze(0)

    embedding,_ = face_transformer(transformed_input.to(device))

    embedding = transform_embedding(embedding)

    return embedding

def get_features(test_list):
    embeddings = []

    for i, img_path in enumerate(test_list):
        embedding = getitems(img_path)
        embeddings.append(embedding)

    return embeddings

def storage_embeddings(embeddings, stoe_root):
    if not os.path.exists(stoe_root):
        os.makedirs(stoe_root)
    for idx in range(len(embeddings)):
        np.save(f'{stoe_root}/{idx}', embeddings[idx])

input_embeddings = get_features(input_paths)
target_embeddings = get_features(target_paths)

storage_embeddings(input_embeddings, '/home/ubuntu/FR_Attack/pixel_stage1/adaface/test/agedb/fake_embedding/')  # real_embeddings
storage_embeddings(target_embeddings, '/home/ubuntu/FR_Attack/pixel_stage1/adaface/test/agedb/real_embedding/')  # real_embeddings
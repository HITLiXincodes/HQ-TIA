import argparse

parser = argparse.ArgumentParser(description='Train face reconstruction network')
parser.add_argument('--FR_system', metavar='<FR_system>', type=str, default='arcface',
                    help='target FR system')
args = parser.parse_args()

import os, sys
sys.path.append(os.getcwd())

import torch
import cv2
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)
# =================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(train=True, device=device, dataset_dir=f'/home/ubuntu/FR_Attack/databases/ffhq/adaface_112')
testing_dataset = MyDataset(train=False, device=device, dataset_dir=f'/home/ubuntu/FR_Attack/databases/ffhq/adaface_112')

train_dataloader = DataLoader(training_dataset, batch_size=48, shuffle=True)
test_dataloader = DataLoader(testing_dataset, batch_size=48, shuffle=False)
# ========================================================

# =================== import Mapping Network =====================
from src.Network import Generator,Discriminator

model_Generator = Generator()
model_Generator.to(device)
model_Discriminator = Discriminator()
model_Discriminator.to(device)
# ========================================================

# =================== import Loss ========================
# ***** face segmentation model
import os
import sys
sys.path.append('/home/ubuntu/FR_Attack/face_parsing')
from model import BiSeNet

def load_parsing():
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    #print(net.device)
    net.to(device)
    save_pth = '/home/ubuntu/FR_Attack/face_parsing/79999_iter.pth'
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net

def masking(images, parsing, face_labels=[1, 2, 3, 4, 5, 10, 11, 12, 13]):
    B, C, H, W = images.shape

    mask = torch.zeros_like(parsing, dtype=torch.bool)
    for label in face_labels:
        mask = mask | (parsing == label)

    mask = mask.unsqueeze(1).expand(-1, C, -1, -1)

    face_images = images * mask.float()
    face_images = F.interpolate(face_images, size=(112, 112), mode='bilinear')

    background_images = images * (~mask).float()
    background_images = F.interpolate(background_images, size=(112, 112), mode='bilinear')

    return face_images, background_images

def center_crop(input_image, crop_ratio=0.75):
    height, width = input_image.shape[2], input_image.shape[3]

    crop_size = int(height * crop_ratio)
    crop_offset = (height - crop_size) // 2
    
    input_cropped = input_image[:, :, crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
    
    return input_cropped

parsing_model = load_parsing()

# ***** Other losses
MAE_loss = torch.nn.L1Loss()
MSE_loss = torch.nn.MSELoss()
BCE_loss = torch.nn.BCELoss()
# ========================================================
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms
# *****************calculate metrics**********************
def get_all_images(dataset_dir):
    img_names = sorted(os.listdir(dataset_dir))
    full_name = []
    for img_name in img_names:
        img_path = os.path.join(dataset_dir, img_name)
        if os.path.isfile(img_path):
            full_name.append(img_path)
    return full_name

def rgb2y_matlab(x):
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y.astype(np.uint8)

def PSNR(im1, im2, use_y_channel=True):
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    mse = np.mean(np.square(im1 - im2)) 
    return 10 * np.log10(255**2 / mse) 

def SSIM(gt_img, noise_img):
    gt_img = rgb2y_matlab(gt_img)
    noise_img = rgb2y_matlab(noise_img)
    ssim_score = compare_ssim(gt_img, noise_img, gaussian_weights=True, 
            sigma=1.5, use_sample_covariance=False)
    return ssim_score

def calmetrics(real_img_paths, fake_img_paths):
    ssim, psnr = [], []
    for i in range(min(len(real_img_paths),len(fake_img_paths))):
        real = cv2.imread(real_img_paths[i])
        fake = cv2.imread(fake_img_paths[i])
        ssim.append(SSIM(real,fake))
        psnr.append(PSNR(real,fake,use_y_channel=True))
    return np.mean(ssim),np.mean(psnr)
# *******************************************************

# =================== Optimizers =========================
# ***** optimizer_Generator
optimizer_Generator = torch.optim.Adam(model_Generator.parameters(), lr=1e-4)
scheduler_Generator = torch.optim.lr_scheduler.StepLR(optimizer_Generator, step_size=50, gamma=0.5)

#optimizer_Discriminator = torch.optim.Adam(filter(lambda p:p.requires_grad, model_Discriminator.parameters()),lr=1e-4)
# ========================================================


# =================== Save models and logs ===============
os.makedirs('training_files', exist_ok=True)
os.makedirs('training_files/models', exist_ok=True)
os.makedirs('training_files/Ground_Truth', exist_ok=True)
os.makedirs('training_files/Generated_images', exist_ok=True)
os.makedirs('training_files/logs_train', exist_ok=True)

with open('training_files/logs_train/log.csv', 'w') as f:
    pass

with open('training_files/logs_train/evaluation.csv', 'w') as f:
    f.write("epoch, SSIM, PSNR\n")

count=1
for embedding, real_image in test_dataloader:
    real_image = real_image.cpu()
    for i in range(real_image.size(0)):
        img = real_image[i].squeeze()
        img = (img+1)/2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(f'training_files/Ground_Truth/{count}.jpg',
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        count+=1
    if (count>=500):
        break
# ======================Train=====================
import torch.nn.functional as F
num_epochs = 100

mean_value = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std_value=torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

for epoch in range(num_epochs):
    model_Generator.train()
    #model_Discriminator.train()

    for embedding, real_image in train_dataloader:
        # ==================forward==================
        fake_image = model_Generator(embedding)
        
        #temp_real = F.interpolate(real_image, size=(224, 224), mode='bilinear')
        #temp_fake = F.interpolate(fake_image, size=(224, 224), mode='bilinear')
        '''
        dis_fake = model_Discriminator(fake_image.detach())
        dis_real = model_Discriminator(real_image)

        dis_loss = 0
        for i in range(len(dis_fake)):

            dis_loss+=(MAE_loss(dis_real[i],dis_fake[i])*(-1))

        model_Discriminator.zero_grad()
        dis_loss.backward()
        optimizer_Discriminator.step()
        '''
        pixel_loss = MAE_loss(real_image, fake_image)

        if epoch>=40:
            real_image_temp = F.interpolate(real_image, size=(512, 512), mode='bilinear')
            fake_image_temp = F.interpolate(fake_image, size=(512, 512), mode='bilinear')

            parse_real = parsing_model(real_image_temp)[0].argmax(1)
            parse_fake = parsing_model(fake_image_temp)[0].argmax(1)

            real_main, _ = masking(real_image_temp, parse_real)
            fake_main, _ = masking(fake_image_temp, parse_fake)

            crop_pixel_loss = MAE_loss(real_main,fake_main)
        else:
            crop_real_image = center_crop(real_image)
            crop_fake_image = center_crop(fake_image)
            crop_pixel_loss = MAE_loss(crop_real_image,crop_fake_image)

        gen_fake = model_Discriminator(fake_image)
        gen_real = model_Discriminator(real_image)

        dis_loss = 0
        for i in range(len(gen_fake)):
            dis_loss+=MSE_loss(gen_real[i],gen_fake[i])
        #dis_loss = lp(real_image,fake_image).mean()

        total_loss = pixel_loss+crop_pixel_loss+dis_loss*0.001
        # ==================backward=================
        optimizer_Generator.zero_grad()
        total_loss.backward()
        optimizer_Generator.step()
        # ==================log======================
    with open('training_files/logs_train/log.csv', 'a') as f:
        f.write(
            f'epoch:{epoch + 1}, \t learning rate: {optimizer_Generator.param_groups[0]["lr"]}, \t pixel_loss:{pixel_loss.data.item()}, \t crop_loss:{crop_pixel_loss.data.item()},\t lp_loss:{dis_loss.data.item()}, \t total_loss:{total_loss.data.item()}\n')

    # ******************** Eval Genrator ********************
    model_Generator.eval()
    count=0
    os.makedirs(f'training_files/Generated_images/epoch_{epoch}', exist_ok=True)
    for embedding, real_image in test_dataloader:
        with torch.no_grad():
            fake_image = model_Generator(embedding).detach().cpu()
        for i in range(fake_image.size(0)):
            img = fake_image[i].squeeze()
            img = (img+1)/2
            im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
            cv2.imwrite(f'training_files/Generated_images/epoch_{epoch}/{count}.jpg',
                        np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
            count+=1
        if (count>=500):
            break

    real_img_paths = get_all_images(f'training_files/Ground_Truth/')
    fake_img_paths = get_all_images(f'training_files/Generated_images/epoch_{epoch}/')
    ssim,psnr = calmetrics(real_img_paths,fake_img_paths)
    with open('training_files/logs_train/evaluation.csv', 'a') as f:
        f.write(
            f"{epoch + 1}, {ssim},{psnr}\n")

    # Save model_Generator
    if epoch%10==0 or epoch>=90:
        torch.save(model_Generator.state_dict(), 'training_files/models/Generator_{}.pth'.format(epoch + 1))

    # Update oprimizer_Generator lr
    scheduler_Generator.step()
# ========================================================

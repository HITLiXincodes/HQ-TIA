import argparse
import os

import torch
import cv2
import numpy as np
from config import add_config_arguments, load_config, resolve_device

parser = argparse.ArgumentParser(description='Train face reconstruction network')
add_config_arguments(parser)
args = parser.parse_args()
config = load_config(args.config, args.set)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = resolve_device(config["train"]["device"])
print("************ NOTE: The torch device is:", device)
# =================== import Dataset ======================
from Dataset import MyDataset
from torch.utils.data import DataLoader

training_dataset = MyDataset(
    train=True,
    device=device,
    dataset_dir=config["paths"]["dataset_dir"],
    train_test_split=config["train"]["train_test_split"],
)
testing_dataset = MyDataset(
    train=False,
    device=device,
    dataset_dir=config["paths"]["dataset_dir"],
    train_test_split=config["train"]["train_test_split"],
)

train_dataloader = DataLoader(
    training_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["num_workers"],
)
test_dataloader = DataLoader(
    testing_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=False,
    num_workers=config["train"]["num_workers"],
)
# ========================================================

# =================== import Mapping Network =====================
from Network import Generator,Discriminator

model_Generator = Generator()
model_Generator.to(device)
model_Discriminator = Discriminator()
model_Discriminator.to(device)
# ========================================================

# =================== import Loss ========================
from loss import ReconstructionLoss

criterion = ReconstructionLoss(config, device)
# ========================================================
from skimage.metrics import structural_similarity as compare_ssim
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
optimizer_Generator = torch.optim.Adam(model_Generator.parameters(), lr=config["train"]["learning_rate"])
scheduler_Generator = torch.optim.lr_scheduler.StepLR(
    optimizer_Generator,
    step_size=config["train"]["lr_step_size"],
    gamma=config["train"]["lr_gamma"],
)
# ========================================================


# =================== Save models and logs ===============
training_output_dir = config["paths"]["training_output_dir"]
models_dir = os.path.join(training_output_dir, "models")
ground_truth_dir = os.path.join(training_output_dir, "Ground_Truth")
generated_images_dir = os.path.join(training_output_dir, "Generated_images")
logs_train_dir = os.path.join(training_output_dir, "logs_train")

os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(ground_truth_dir, exist_ok=True)
os.makedirs(generated_images_dir, exist_ok=True)
os.makedirs(logs_train_dir, exist_ok=True)

with open(os.path.join(logs_train_dir, "log.csv"), 'w') as f:
    f.write("")

with open(os.path.join(logs_train_dir, "evaluation.csv"), 'w') as f:
    f.write("epoch, SSIM, PSNR\n")

count=1
for embedding, real_image in test_dataloader:
    real_image = real_image.cpu()
    for i in range(real_image.size(0)):
        img = real_image[i].squeeze()
        img = (img+1)/2
        im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(os.path.join(ground_truth_dir, f"{count}.jpg"),
                    np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
        count+=1
    if (count >= config["train"]["eval_image_limit"]):
        break
# ======================Train=====================
num_epochs = config["train"]["num_epochs"]

for epoch in range(num_epochs):
    model_Generator.train()

    for embedding, real_image in train_dataloader:
        # ==================forward==================
        fake_image = model_Generator(embedding)

        losses = criterion(real_image, fake_image, model_Discriminator, epoch)
        pixel_loss = losses["pixel_loss"]
        crop_pixel_loss = losses["crop_pixel_loss"]
        dis_loss = losses["dis_loss"]
        total_loss = losses["total_loss"]
        # ==================backward=================
        optimizer_Generator.zero_grad()
        total_loss.backward()
        optimizer_Generator.step()
        # ==================log======================
    with open(os.path.join(logs_train_dir, "log.csv"), 'a') as f:
        f.write(
            f'epoch:{epoch + 1}, \t learning rate: {optimizer_Generator.param_groups[0]["lr"]}, \t pixel_loss:{pixel_loss.data.item()}, \t crop_loss:{crop_pixel_loss.data.item()},\t lp_loss:{dis_loss.data.item()}, \t total_loss:{total_loss.data.item()}\n')

    # ******************** Eval Genrator ********************
    model_Generator.eval()
    count=0
    epoch_generated_dir = os.path.join(generated_images_dir, f"epoch_{epoch}")
    os.makedirs(epoch_generated_dir, exist_ok=True)
    for embedding, real_image in test_dataloader:
        with torch.no_grad():
            fake_image = model_Generator(embedding).detach().cpu()
        for i in range(fake_image.size(0)):
            img = fake_image[i].squeeze()
            img = (img+1)/2
            im = (img.numpy().transpose(1, 2, 0) * 255).astype(int)
            cv2.imwrite(os.path.join(epoch_generated_dir, f"{count}.jpg"),
                        np.array([im[:, :, 2], im[:, :, 1], im[:, :, 0]]).transpose(1, 2, 0))
            count+=1
        if (count >= config["train"]["eval_image_limit"]):
            break

    real_img_paths = get_all_images(ground_truth_dir)
    fake_img_paths = get_all_images(epoch_generated_dir)
    ssim,psnr = calmetrics(real_img_paths,fake_img_paths)
    with open(os.path.join(logs_train_dir, "evaluation.csv"), 'a') as f:
        f.write(
            f"{epoch + 1}, {ssim},{psnr}\n")

    # Save model_Generator
    if epoch % config["train"]["model_save_interval"] == 0 or epoch >= config["train"]["save_all_after_epoch"]:
        torch.save(model_Generator.state_dict(), os.path.join(models_dir, "Generator_{}.pth".format(epoch + 1)))

    # Update optimizer_Generator lr
    scheduler_Generator.step()
# ========================================================

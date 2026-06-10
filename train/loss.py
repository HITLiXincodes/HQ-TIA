import sys

import torch
import torch.nn.functional as F

try:
    from config import require_config_path
except ModuleNotFoundError:
    from .config import require_config_path


class ReconstructionLoss:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.mae_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.parsing_model = self._load_parsing()

    def _load_parsing(self):
        sys.path.append(require_config_path(self.config, "paths.face_parsing_dir"))
        from model import BiSeNet

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.to(self.device)
        save_pth = require_config_path(self.config, "paths.face_parsing_checkpoint")
        net.load_state_dict(torch.load(save_pth))
        net.eval()
        return net

    def _masking(self, images, parsing, face_labels=(1, 2, 3, 4, 5, 10, 11, 12, 13)):
        _, channels, _, _ = images.shape

        mask = torch.zeros_like(parsing, dtype=torch.bool)
        for label in face_labels:
            mask = mask | (parsing == label)

        mask = mask.unsqueeze(1).expand(-1, channels, -1, -1)

        face_images = images * mask.float()
        face_images = F.interpolate(face_images, size=(112, 112), mode="bilinear")

        background_images = images * (~mask).float()
        background_images = F.interpolate(background_images, size=(112, 112), mode="bilinear")

        return face_images, background_images

    def _center_crop(self, input_image):
        crop_ratio = self.config["train"]["crop_ratio"]
        height = input_image.shape[2]

        crop_size = int(height * crop_ratio)
        crop_offset = (height - crop_size) // 2

        input_cropped = input_image[
            :,
            :,
            crop_offset:crop_offset + crop_size,
            crop_offset:crop_offset + crop_size,
        ]

        return input_cropped

    def _crop_loss(self, real_image, fake_image, epoch):
        if epoch >= self.config["train"]["face_parse_start_epoch"]:
            real_image_temp = F.interpolate(real_image, size=(512, 512), mode="bilinear")
            fake_image_temp = F.interpolate(fake_image, size=(512, 512), mode="bilinear")

            parse_real = self.parsing_model(real_image_temp)[0].argmax(1)
            parse_fake = self.parsing_model(fake_image_temp)[0].argmax(1)

            real_main, _ = self._masking(real_image_temp, parse_real)
            fake_main, _ = self._masking(fake_image_temp, parse_fake)

            return self.mae_loss(real_main, fake_main)

        crop_real_image = self._center_crop(real_image)
        crop_fake_image = self._center_crop(fake_image)
        return self.mae_loss(crop_real_image, crop_fake_image)

    def _perceptual_loss(self, real_features, fake_features):
        dis_loss = 0
        for i in range(len(fake_features)):
            dis_loss += self.mse_loss(real_features[i], fake_features[i])
        return dis_loss

    def __call__(self, real_image, fake_image, discriminator, epoch):
        pixel_loss = self.mae_loss(real_image, fake_image)
        crop_pixel_loss = self._crop_loss(real_image, fake_image, epoch)

        gen_fake = discriminator(fake_image)
        gen_real = discriminator(real_image)
        dis_loss = self._perceptual_loss(gen_real, gen_fake)

        total_loss = (
            pixel_loss
            + crop_pixel_loss
            + dis_loss * self.config["train"]["perceptual_loss_weight"]
        )

        return {
            "pixel_loss": pixel_loss,
            "crop_pixel_loss": crop_pixel_loss,
            "dis_loss": dis_loss,
            "total_loss": total_loss,
        }

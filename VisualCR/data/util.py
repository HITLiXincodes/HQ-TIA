import os
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = {'.jpg', '.png'}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS


def get_paths_from_images(path):
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    return sorted(images)


def image_to_tensor(img, min_max=(-1, 1)):
    tensor = TF.to_tensor(img)
    return tensor * (min_max[1] - min_max[0]) + min_max[0]

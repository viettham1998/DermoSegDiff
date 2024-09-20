import os
import shutil
import random

folder_a = '/home/hvtham/Selab/datasets/isic2016/train/images'
folder_b = '/home/hvtham/Selab/datasets/isic2016/train/masks'
folder_c = '/home/hvtham/Selab/datasets/isic2016/val/images'
folder_d = '/home/hvtham/Selab/datasets/isic2016/val/masks'

os.makedirs(folder_c, exist_ok=True)
os.makedirs(folder_d, exist_ok=True)

images_a = os.listdir(folder_a)

selected_images = random.sample(images_a, 150)

for image in selected_images:
    shutil.move(os.path.join(folder_a, image), os.path.join(folder_c, image))

    mask_name = image.replace('.jpg', '_Segmentation.png')
    shutil.move(os.path.join(folder_b, mask_name), os.path.join(folder_d, mask_name))

print("Đã di chuyển 150 ảnh và mask tương ứng.")

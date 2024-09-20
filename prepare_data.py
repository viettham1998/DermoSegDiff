import os
import shutil
import random

def split_train_val_isic_2016(images_src, masks_src, images_des, masks_des):
    os.makedirs(images_des, exist_ok=True)
    os.makedirs(masks_des, exist_ok=True)
    list_images = os.listdir(images_src)
    selected_images = random.sample(list_images, 150)
    for image in selected_images:
        shutil.move(os.path.join(images_src, image), os.path.join(images_des, image))
        mask_name = image.replace('.jpg', '_Segmentation.png')
        shutil.move(os.path.join(masks_src, mask_name), os.path.join(masks_des, mask_name))

    print("Done!!!")

def move_superpixels_isic_2017(images_src, images_des):
    os.makedirs(images_des, exist_ok=True)
    files_all = os.listdir(images_src)
    for file in files_all:
        if 'superpixels' in file:
            shutil.move(os.path.join(images_src, file), os.path.join(images_des, file))


def main():
    #train_images = '/home/hvtham/Selab/datasets/isic2016/train/images'
    #train_masks = '/home/hvtham/Selab/datasets/isic2016/train/masks'
    #val_images = '/home/hvtham/Selab/datasets/isic2016/val/images'
    #val_masks = '/home/hvtham/Selab/datasets/isic2016/val/masks'
    #split_train_val_isic_2016(train_images,train_masks,val_images, val_masks)

    train_images_2017 = '/home/hvtham/Selab/datasets/isic2017/train/images'
    superpixels_images_2017 = '/home/hvtham/Selab/datasets/isic2017/train/superpixels'
    move_superpixels_isic_2017(train_images_2017, superpixels_images_2017)
import os
import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pyvips
import warnings
import random
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from random import randrange
from pathlib import Path
from glob import glob
from skimage.exposure import is_low_contrast
from scipy.ndimage import zoom, rotate
from skimage.io import imread, imsave
from collections import defaultdict
from openslide import OpenSlide

class ImageProcessor:
    def __init__(self, data_path='../input/mayo-clinic-strip-ai/', IMG_SIZE=224, IMG_CHANNELS=3):
        self.data_path = data_path
        self.IMG_SIZE = IMG_SIZE
        self.IMG_CHANNELS = IMG_CHANNELS

    def setup(self):
        random.seed = 19
        self.train_path = os.path.join(self.data_path, 'train/')
        self.train_label_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        self.train_images_data = glob(os.path.join(self.train_path, "*"))
        print(f"Number of images in a training set: {len(self.train_images_data)}")

    def get_img_info(self):
        return get_img_info(self.train_images_data, self.train_label_df)

    def process_single_tile(self, row, output_folder):
        exists_tiles = os.path.exists(output_folder)

        if not exists_tiles:
            print('creating folder', output_folder)
            os.makedirs(output_folder)

        last_img_index = 0
        print('started processing image:', row['image_id'])
        last_img_index = self.split_save_tiles(row, last_img_index, output_folder)
        print('processed image:', row['image_id'], 'Total processed tiles:', last_img_index)

    def split_save_tiles(self, row, last_img_index, path_tiles):
        image_id = row['image_id']

        width = row['width']
        height = row['height']
        big_dim = row['big_dim']
        split_size = row['split_size']
        input_train_path = row['path']
        label = row['label']
        center_id = row['center_id']

        n_across = 1
        n_down = 1

        vips_img = pyvips.Image.new_from_file(input_train_path, access='sequential')
        plt.imshow(vips_img)
        plt.title(f"Original Image")
        plt.show()
        if split_size == 1:
            crop_width = width
            crop_height = height
        elif big_dim == 'width':
            crop_width = width // split_size
            crop_height = height
            n_across = split_size
        else:
            crop_height = height // split_size
            crop_width = width
            n_down = split_size

        for x in range(n_across):
            for y in range(n_down):
                vips_tile = None
                if split_size > 1:
                    vips_tile = vips_img.crop(x * crop_width, y * crop_height, crop_width, crop_height)
                else:
                    vips_tile = vips_img

                # Display the original tile
                plt.imshow(vips_tile)
                plt.title(f"Image Tile After Cropping: {image_id}_{x + y}")
                plt.show()
                print(last_img_index, image_id, 'processing image with splits(', split_size, ')', crop_width,
                      ' X ', crop_height)
                vips_tile = vips_tile.thumbnail_image(self.IMG_SIZE, height=self.IMG_SIZE, size='force')
                if is_low_contrast(vips_tile):
                    print('low contrast tile can not be saved ')
                    continue
                tile_name = image_id + '_' + str(x + y)
                self.save_tile(path_tiles, tile_name, vips_tile, label)

                last_img_index += 1

        vips_img = None
        return last_img_index

    def save_tile(self, path, name, vips_tile, label):
        img = vips_tile.numpy()
        im = Image.fromarray(img)
        im.save(os.path.join(path, name + '_0.tif'))
        plt.imshow(im)
        plt.title(f"Processed Image: {name}_0")
        plt.show()

        im_rotated_90 = im.rotate(90)
        im_rotated_90.save(os.path.join(path, name + '_1.tif'))
        plt.imshow(im_rotated_90)
        plt.title(f"Processed Image: {name}_1 (Rotated 90 degrees)")
        plt.show()

        if label == 'LAA':
            im_rotated_135 = Image.fromarray(rotate(img, 135, reshape=False, mode='reflect'))
            im_rotated_135.save(os.path.join(path, name + '_2.tif'))
            plt.imshow(im_rotated_135)
            plt.title(f"Processed Image: {name}_2 (Rotated 135 degrees)")
            plt.show()

            im_rotated_180 = im_rotated_90.rotate(90)
            im_rotated_180.save(os.path.join(path, name + '_3.tif'))
            plt.imshow(im_rotated_180)
            plt.title(f"Processed Image: {name}_3 (Rotated 180 degrees)")
            plt.show()

            im_rotated_225 = Image.fromarray(rotate(img, 225, reshape=False, mode='reflect'))
            im_rotated_225.save(os.path.join(path, name + '_4.tif'))
            plt.imshow(im_rotated_225)
            plt.title(f"Processed Image: {name}_4 (Rotated 225 degrees)")
            plt.show()

        print(name, 'done saving a tile...')

def main():
    image_processor = ImageProcessor()
    image_processor.setup()

    image_info = image_processor.get_img_info()

    # Example: Process a single tile for the specified image_id
    image_id_to_find = '00c058_0'
    single_row = image_info[image_info['image_id'] == image_id_to_find]
    if not single_row.empty:
        image_processor.process_single_tile(single_row.iloc[0], 'train/processed_image/')

if __name__ == "__main__":
    main()
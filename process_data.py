import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

"""
convert the image to the specified dimension - img_width*img_height
"""
def resize_image(img_width, img_height, image_path):
    image = Image.open(image_path)
    image = image.resize((img_height,img_width), Image.ANTIALIAS)
    path = image_path.split('/')
    file_name = path[len(path)-1]
    path.pop(len(path)-1)
    save_dir = "/".join(path)
    save_dir = save_dir+"/resized/"
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    print(save_dir)
    image.save(save_dir+file_name,quality=95)

"""
resize all images under a direcoty (dir), with dim -- img_width,img_height
"""
def resige_images(dir, img_width, img_height):
    print(dir)
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(dir,file)
        if os.path.isdir(path):
            resige_images(path, img_width, img_height)
        else:
            resize_image(img_width,img_height,path)

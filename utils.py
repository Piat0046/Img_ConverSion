from detectron2.structures import polygons_to_bitmask
from PIL import Image

import os
import numpy as np
import pycocotools.mask as mask_util 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. '+ directory)

def convert_cmap(mask_img, img_w, img_h ):
    # 01: black smoke [0,15,255]
    # 02: gray smoke [130,130,130]
    # 03: white smoke [255,255,255]
    # 04: fire smoke [255,0,0]
    cmap = [[0,0,0],[0,15,255],[130,130,130],[255,255,255],[255,0,0]]
    palette = [0,0,0,0,15,255,130,130,130,255,255,255,255,0,0]

    img_png = np.zeros((img_w, img_h), np.uint8)

    for index, val_col in enumerate(cmap):
        img_png[np.where(np.all(mask_img == val_col, axis=-1))] = index

    img_png = Image.fromarray(img_png).convert('P')
    # Palette information injection
    img_png.putpalette(palette)

    return img_png

def make_json(name, dic, path):
    json_path = os.path.join(path, name+'json')
    with open(json_path, "w") as json_file:
        json.dump(dic, json_file)



def check_color(text):
    class_num = text
    if class_num == '01':
        return [255,15,0]
    elif class_num == '02':
        return [130,130,130]
    elif class_num == '03':
        return [255,255,255]
    elif class_num == '04':
        return [0,0,255]
    
# Fast run length encoding
def rle (img):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    
    return starts_ix, lengths



def polygon_to_rle(polygon: list, shape=(1920, 1080)):
    '''
    polygon: a list of [x1, y1, x2, y2,....]
    shape: shape of bitmask
    Return: RLE type of mask
    '''
    poly = []
    for i in polygon:  poly.extend(i)
    mask = polygons_to_bitmask([np.asarray(poly) + 0.25], shape[0], shape[1]) # add 0.25 can keep the pixels before and after the conversion unchanged
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle

def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin

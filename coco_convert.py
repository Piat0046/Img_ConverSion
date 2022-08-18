import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool
from os.path import join as path_join
from utils import *


def main(name):
    try:
        global path
        global mask_save_path
        global jpg_save_path
        global images
        global annotations
        
        # load files & folder
        img_path = os.path.join(path, 'img')
        json_path = os.path.join(path, 'label')
        img_file = os.path.join(img_path, name+'.jpg')
        json_file = os.path.join(json_path, name+'.json')
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        with open(json_file, 'r', encoding='utf-8-sig') as f:
            json_dic = json.load(f)
        
        # make poly mask
        img_w, img_h = img.shape[0], img.shape[1]
        annotations = []
        for num in range(len(json_dic['annotations'])):
            mask_zero = np.zeros((img_w, img_h,3), dtype = np.uint8)
            class_num = json_dic['annotations'][num]['class']
            color = check_color(class_num)
            if 'polygon' in list(json_dic['annotations'][num].keys()):
                polygon = json_dic['annotations'][num]['polygon']

            else:
                x, y, w, h = json_dic['annotations'][num]['box']
                polygon = [[x,y],[w, y], [w, h], [x, h]]
            poly_mask = np.array(polygon, np.int32)
            cv2.fillPoly(mask_zero, [poly_mask], color)
            img_gray = cv2.cvtColor(mask_zero, cv2.COLOR_BGR2GRAY)
            _ , mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY) 
            rle = polygon_to_rle(polygon, (img_w, img_h))
            rle['counts'] = rle['counts'].decode('ascii')
            annotations.append({
                    'category_id': int(class_num),
                    'segmentation': rle,
                    'area': float(mask.sum()),
                    'bbox': [int(x) for x in mask2bbox(mask)],
                    'iscrowd': 0
                })
            #print(type(annotations[0]['segmentation']['counts']))
        images = {
            'width': img.shape[1],
            'height': img.shape[0],
            'file_name': name
        }
        return annotations, images
            
    except Exception as e:
        print(e)



if __name__=='__main__':

    path = '/home/ubuntu/fire/aihub/Validation'
    jpg_path = os.path.join(path, 'img')
    img_list = [i.rsplit('.')[0] for i in os.listdir(jpg_path)]

    # Make Folder
    mode = 'valid'
    save_path = '/home/ubuntu/fire/aihub/datase'
    folder_list = [mode, mode+'/Annotations', mode+'/JPEGImages']
    folder_path_list = [os.path.join(save_path, folder) for folder in folder_list]
    jpg_save_path = folder_path_list[2]
    mask_save_path = folder_path_list[1]
    for folder_path in folder_path_list: createFolder(folder_path)

    image_id = 1
    ann_id   = 1
    images = []
    annotations = []

    cpu = 16
    with Pool(cpu) as pool:
        for num, i in enumerate(tqdm(pool.imap_unordered(main, img_list))):
            annotation, image = i
            for ann in annotation:
                ann['id'] = ann_id
                ann['image_id'] =  image_id
                annotations.append(ann)
                ann_id += 1
            image['id'] = image_id
            images.append(image)
            image_id += 1
            

    json_path = path_join(path_join(save_path, mode), mode+'.json')

    info = {
        'year': 2012,
        'version': 1,
        'description': 'Pascal SBD',
    }

    categories = [{'id': x+1} for x in range(4)]

    with open(json_path, 'w') as f:
        json.dump({
            'info': info,
            'images': images,
            'annotations': annotations,
            'licenses': {},
            'categories': categories
        }, f)

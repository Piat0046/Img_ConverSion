import os
import sys
import cv2
import json
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool
from os.path import join as path_join
from utils import *

def main(name):
    try:
        global path
        
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
        image = {
            'width': img.shape[1],
            'height': img.shape[0],
            'file_name': name+'.jpg'
        }
        return annotations, image
            
    except Exception as e:
        print(e)
        print(name)
        return None



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(
    description='Make Aihub dataset to coco dataset')

    parser.add_argument('--path', type=str, required=True,
                        help='Custom dataset path')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Select your dataset')

    args = parser.parse_args()

    path = args.path
    jpg_path = os.path.join(path, 'img')
    img_list = [i.rsplit('.')[0] for i in os.listdir(jpg_path)]

    # Make Folder
    mode = args.mode

    image_id = 1
    ann_id   = 1
    videos = []
    annotations = []
    print(len(img_list))
    cpu = 16
    with Pool(cpu) as pool:
        for num, i in enumerate(tqdm(pool.imap_unordered(main, img_list),total=len(img_list))):
            if not i == None:
                annotation, image = i
                for ann in annotation:
                    ann['id'] = ann_id
                    ann['video_id'] =  image_id
                    annotations.append(ann)
                    ann_id += 1
                image['id'] = image_id
                videos.append(image)
                image_id += 1
            

    json_path = path_join(path, mode+'.json')

    info = {
        'year': 2012,
        'version': 1,
        'description': 'Pascal SBD',
    }

    categories = [{"id": 1, "name":"black smoke"}, {"id": 2,"name":"gray smoke"}, {"id": 3,"name":"white smoke"}, {"id": 4,"name":"fire"}]

    with open(json_path, 'w') as f:
        json.dump({
            'info': info,
            'videos': videos,
            'annotations': annotations,
            'licenses': {},
            'categories': categories
        }, f)

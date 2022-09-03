import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool
from os.path import join as path_join
from utils import createFolder, convert_cmap, check_color


def main(name):
    try:
        global path
        global mask_save_path
        global jpg_save_path
        global image_id
        global ann_id
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

        # ready json data
        img_meta = {"folder": name[:-5],
                    "frame": name,
                    "object": []
                    }

        # make poly mask
        img_w, img_h = img.shape[0], img.shape[1]
        annotations = json_dic['annotations']
        mask_zero = np.zeros((img_w, img_h,3), dtype = np.uint8)
        for num in range(len(annotations)):
            class_num = annotations[num]['class']
            img_meta['object'].append(int(class_num))
            color = check_color(class_num)
            if 'polygon' in list(annotations[num].keys()):
                polygon = json_dic['annotations'][num]['polygon']

            else:
                x, y, w, h = json_dic['annotations'][num]['box']
                polygon = [[x,y],[w, y], [w, h], [x, h]]
            poly_mask = np.array(polygon, np.int32)
            cv2.fillPoly(mask_zero, [poly_mask], color)
       
        mask = cv2.cvtColor(mask_zero, cv2.COLOR_BGR2RGB)
        img_png = convert_cmap(mask, img_w, img_h)

        # Save img
        mask_path = path_join(mask_save_path, name[:-5])
        jpg_path = path_join(jpg_save_path, name[:-5])

        img_png.save(path_join(mask_path, f'{name}.png'))
        cv2.imwrite(path_join(jpg_path, f'{name}.jpg'), img)
        
        return img_meta
            
    except Exception as e:
        print(e)



if __name__=='__main__':

    path = '/home/ubuntu/fire/data/Training'
    jpg_path = os.path.join(path, 'img')
    img_list = [i.rsplit('.')[0] for i in os.listdir(jpg_path)]

    # Make Folder
    mode = 'train'
    save_path = '/home/ubuntu/fire/fire_vos'
    folder_list = [mode, mode+'/Annotations', mode+'/JPEGImages']
    folder_name = set([i[:-5] for i in img_list])
    for i in folder_name: folder_list.append(path_join(folder_list[1], i)), folder_list.append(path_join(folder_list[2], i))
    folder_path_list = [os.path.join(save_path, folder) for folder in folder_list]
    jpg_save_path = folder_path_list[2]
    mask_save_path = folder_path_list[1]
    for folder_path in folder_path_list: createFolder(folder_path)
    json_data = {}
    for i in folder_name:
        json_data[i] = {'object' : {1:{"category": "black_smoke", 
                                       "frames": []},
                                    2:{"category": "gray_smoke", 
                                       "frames": []},
                                    3:{"category": "white_smoke", 
                                       "frames": []},
                                    4:{"category": "fire", 
                                       "frames": []}}}

    cpu = 10
    with Pool(cpu) as pool:
        for num, i in enumerate(tqdm(pool.imap_unordered(main, img_list))):
            try:
                for b in range(len(i['object'])):
                    json_data[i['folder']]['object'][i['object'][b]]['frames'].append(i['frame'])
            except Exception as e:
                print(i)
                print(e)
        
    json_path = path_join(path_join(save_path, mode), mode+'.json')

    with open(json_path, 'w') as f:
        json.dump(json_data, f)

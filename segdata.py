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
        annotations = json_dic['annotations']
        mask_zero = np.zeros((img_w, img_h,3), dtype = np.uint8)
        #"license": 1,"file_name": "COCO_test2014_000000027454.jpg","coco_url": "http://images.cocodataset.org/test2014/COCO_test2014_000000027454.jpg","height": 640,"width": 427,"date_captured": "2013-11-22 01:57:00","id": 27454}
        for num in range(len(annotations)):
            class_num = annotations[num]['class']
            color = check_color(class_num)
            if 'polygon' in list(annotations[num].keys()):
                polygon = json_dic['annotations'][num]['polygon']

            else:
                x, y, w, h = json_dic['annotations'][num]['box']
                polygon = [[x,y],[w, y], [w, h], [x, h]]
            print(polygon)
            print(img_w, img_h)
            poly_mask = np.array(polygon, np.int32)
            cv2.fillPoly(mask_zero, [poly_mask], color)
       
        # write json data
        img_meta = {"license": 1,
                     "file_name": img_file,
                     "coco_url": "http://images.cocodataset.org/test2014/COCO_test2014_000000027454.jpg",
                     "height": img_w, "width": img_h,
                     "date_captured": json_dic['image']['date']
                    }

        mask = cv2.cvtColor(mask_zero, cv2.COLOR_BGR2RGB)
        img_png = convert_cmap(mask, img_w, img_h)

        # Save img
        #mask_path = path_join(mask_save_path, name[:-5])
        #jpg_path = path_join(jpg_save_path, name[:-5])

        img_png.save(path_join(mask_save_path, f'{name}.png'))
        cv2.imwrite(path_join(jpg_save_path, f'{name}.jpg'), img)

        return img_meta
            
    except Exception as e:
        print(e)



if __name__=='__main__':

    path = '/home/ubuntu/fire/aihub/Validation'
    jpg_path = os.path.join(path, 'img')
    img_list = [i.rsplit('.')[0] for i in os.listdir(jpg_path)]

    # Make Folder
    mode = 'valid'
    save_path = '/home/ubuntu/fire/aihub/dataset'
    folder_list = [mode, mode+'/Annotations', mode+'/JPEGImages']
    #folder_name = set([i[:-5] for i in img_list])
    #for i in folder_name: folder_list.append(path_join(folder_list[1], i)), folder_list.append(path_join(folder_list[2], i))
    folder_path_list = [os.path.join(save_path, folder) for folder in folder_list]
    jpg_save_path = folder_path_list[2]
    mask_save_path = folder_path_list[1]
    for folder_path in folder_path_list: createFolder(folder_path)

    make_json = {"info": {"description": "COCO 2014 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2014,"contributor": "COCO Consortium","date_created": "2017/09/01"},
                 "images" : [],
                 "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},{"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},{"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},{"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},{"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},{"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}],
                 "categories" : [{"supercategory": "fire","id": 1,"name": "black_smoke"},{"supercategory": "fire","id": 2,"name": "gray_smoke"},{"supercategory": "fire","id": 3,"name": "white_smoke"},{"supercategory": "fire","id": 4,"name": "fire"}]
                 }


    cpu = 16
    with Pool(cpu) as pool:
        for num, i in enumerate(tqdm(pool.imap_unordered(main, img_list[:1]))):
            i['id'] = num
            make_json['images'].append(i)
    
    json_path = path_join(path_join(save_path, mode), mode+'.json')

    with open(json_path, "w") as json_file:
        json.dump(make_json, json_file)

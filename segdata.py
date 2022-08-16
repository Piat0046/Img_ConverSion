import os
import cv2
import json
import numpy as np
from utils import createFolder
from pprint import pprint

# Make Save Folder
mode = 'vaild'
save_path = '/home/ubuntu/fire/aihub/dataset'
folder_list = [mode, mode+'/Annotations', mode+'/JPEGImages']
jpg_save_path = folder_list[2]
mask_save_path = folder_list[1]
folder_path_list = [os.path.join(save_path, folder) for folder in folder_list]
for folder_path in folder_path_list: createFolder(folder_path)


path = '/home/ubuntu/fire/aihub/Validation'

img_list = os.listdir(path)

def convert_cmap(mask_img):
    cmap = [[0,0,0],[250,255,100]]
    palette = [0,0,0,250,255,100]

    img_png = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for index, val_col in enumerate(cmap):
        img_png[np.where(np.all(mask_img == val_col, axis=-1))] = index

    img_png = Image.fromarray(img_png).convert('P')
    # Palette information injection
    img_png.putpalette(palette)

    return img_png


def main(img):
    global path

    # load files
    name = img
    img_path = os.path.join(path, 'img')
    json_path = os.path.join(path, 'label')
    img_file = os.path.join(img_path, img+'.jpg')
    json_file = os.path.join(json_path, img+'.json')
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    with open(json_file, 'r', encoding='utf-8-sig') as f:
        json_dic = json.load(f)
    
    # make poly mask
    img_w, img_h = img.shape[0], img.shape[1]
    polygon = json_dic['annotations'][0]['polygon']

    color = [100,255,250] # yellow
    mask_zero = np.zeros((img_w, img_h,3), dtype = np.uint8)
    poly_mask = np.array(polygon, np.int32)
    cv2.fillPoly(mask_zero, [poly_mask], color)
    mask = cv2.cvtColor(mask_zero, cv2.COLOR_BGR2RGB)
    img_png = convert_cmap(mask)

    img_png.save(os.path.join(mask_path, f'{file_name}_{num}.png'))
    cv2.imwrite(os.path.join(jpg_path, f'{file_name}_{num}.jpg'), img)
    # train_txt.write(f'{file_name}_{num}\n')


    return 

img_2 = 'S3-N1215MF00644.jpg'


pprint(main(img_2.rsplit('.')[0]))

# if __name__=='__main__':

#     path = r'/home/ubuntu/Cardata/scratch'
#     with open('./labels/train2.txt', 'w', encoding='UTF=8') as f:
#         f.close()

#     all_files_list = os.listdir(path)
#     files_list = [i for i in all_files_list if 'png' not in i and 'tif' not in i]
#     cpu = 16
#     with Pool(cpu) as pool:
#         for _ in tqdm(pool.imap_unordered(main, files_list)):
#             pass
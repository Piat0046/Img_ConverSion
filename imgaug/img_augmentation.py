from random import choice
import cv2
import argparse
from tqdm import tqdm
from imgaug.augmenters.blur import MotionBlur
from imgaug.augmenters.meta import OneOf
import numpy as np
import imgaug as ia
import imgaug.augmenters as augmenter
import imgaug.augmenters.imgcorruptlike as cor
import os
import shutil

def mask_write():
    pass

def new_imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=" "
    )
    parser.add_argument(
        "--img_dir", type=str, help="The img directory", required=True
    )
    parser.add_argument(
        "--mask_dir", type=str, help="The mask directory", required=True
    )
    parser.add_argument(
        "--txt_dir", type=str, help="txt directory", default= None
    )

    parser.add_argument(
        "--save_dir", type=str, help="Save directory", default= None
    )
    return parser.parse_args()


def main():


    opt = parse_arguments()

    ####### img path
    img_dir = opt.img_dir
    mask_dir = opt.mask_dir
    ####### txt path
    if opt.txt_dir == None:
        txt_path = os.path.join(img_dir, "labels.txt")
    else:
        txt_path = opt.txt_dir

    ####### new img dir
    if opt.save_dir == None:
        img_save_dir = img_dir + "_aug"
    else:
        img_save_dir = opt.save_dir
    os.makedirs(img_save_dir, exist_ok=True)


    sometimes = lambda aug: augmenter.Sometimes(0.8, aug)
    sometimes_low = lambda aug: augmenter.Sometimes(0.3, aug)


    # rf = open(txt_path, 'r', encoding='utf-8')
    # infos = rf.readlines()
    # rf.close()
    # wf = open(txt_path, 'a', encoding='utf-8')
    # length = len(infos)
    files = os.listdir(img_dir)

    wf = open('/home/ubuntu/Cardata/labels/train_aug2.txt', 'w', encoding='utf-8')

    for file in tqdm(files): # per line
        if 'aug' in file:
            continue
        try:
            num = choice([0,1,2,3])
            file_name = file.rsplit('.')[0]
            jpg_name = file_name+'.jpg'
            mask_name = file_name+'.png'

            jpg_path = img_dir
            mask_path = mask_dir

            jpg_save_path = os.path.join('/home/ubuntu/Cardata/jpg_crop',file_name+'_aug2.jpg')
            mask_save_path = os.path.join('/home/ubuntu/Cardata/mask_crop',file_name+'_aug2.png')

            img = os.path.join(jpg_path, jpg_name)
            mask = os.path.join(mask_path, mask_name)
            
            ff = np.fromfile(img, np.uint8)
            img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

            h, w, _ = img.shape
            x1 = 0
            y1 = 0
            x2 = w
            y2 = h
            
            input_img = img[np.newaxis, :, :, :]
            bbox = [ia.BoundingBox(x1 = x1, y1 = y1, x2 = x2, y2 = y2)]


            ####### aug seq 설정

            seq = augmenter.Sequential([
                #OneOf([augmenter.GaussianBlur(sigma=(0,2)), cor.Pixelate(severity=(1,2))]),
                #augmenter.pillike.EnhanceSharpness(factor=3.0),
                #augmenter.Sharpen(alpha=(0.1), lightness=(0)),
                #augmenter.Grayscale(alpha=(0.7)),
                #augmenter.SigmoidContrast(gain=(0), cutoff=(0.5)),
                #augmenter.HistogramEqualization(),
                #augmenter.LogContrast(gain=(0.6)),
                #sometimes(augmenter.Rot90((1, 3)))
                #augmenter.pillike.Autocontrast((5), per_channel=True),
                #augmenter.UniformColorQuantization(n_colors=(8)),
                #augmenter.pillike.FilterEdgeEnhanceMore()
                augmenter.Snowflakes(flake_size=(0.4), speed=(0.005))
                ],
                # sometimes_low(cor.Spatter(severity=(3,4))),
                # sometimes(cor.Pixelate(severity=(1,2)))],
                # sometimes(augmenter.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)),
                # sometimes(augmenter.TranslateX(percent=(-0.02, 0.02))),
                # sometimes(augmenter.TranslateY(percent=(-0.02, 0.02))),
                #random_order=True # 조합 순서를 무작위로
                )
            
            aug_img, aug_bbox = seq(images = input_img, bounding_boxes = bbox)
            
            # 이미지 저장
            
            new_imwrite(jpg_save_path, aug_img[0])

            shutil.copy(mask, mask_save_path)

            # INFO TXT 파일 생성
            # data = name + '_aug.jpg' + ' ' + string
            wf.write(file_name+'_aug2'+'\n')
            
        except Exception as e:
            print(e)
            pass

    # wf.close()


if __name__ == "__main__":
    main()
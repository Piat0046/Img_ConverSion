import os
import numpy as np
from PIL import Image

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


from detectron2.structures import polygons_to_bitmask
def polygon_to_rle(polygon: list, shape=(1920, 1080)):
    '''
    polygon: a list of [x1, y1, x2, y2,....]
    shape: shape of bitmask
    Return: RLE type of mask
    '''
    mask = polygons_to_bitmask([np.asarray(polygon) + 0.25], shape[0], shape[1]) # add 0.25 can keep the pixels before and after the conversion unchanged
    rle = mask_util.encode(np.asfortranarray(mask))
    return rle

polygon = [[757, 39], [725, 51], [702, 59], [679, 66], [628, 98], [589, 121], [558, 128], [521, 159], [513, 185], [505, 214], [502, 235], [498, 262], [509, 296], [537, 318], [562, 339], [576, 347], [596, 352], [626, 356], [632, 367], [636, 392], [654, 415], [655, 437], [657, 461], [657, 487], [665, 504], [680, 515], [700, 541], [703, 570], [715, 595], [721, 611], [733, 636], [746, 652], [742, 670], [731, 687], [720, 710], [718, 730], [724, 747], [728, 766], [731, 784], [735, 803], [739, 821], [740, 843], [754, 862], [765, 878], [773, 898], [781, 908], [792, 919], [820, 921], [854, 921], [881, 919], [918, 918], [947, 919], [980, 921], [1011, 921], [1031, 918], [1059, 895], [1065, 877], [1065, 856], [1062, 835], [1055, 817], [1047, 803], [1046, 780], [1047, 754], [1050, 733], [1050, 711], [1047, 689], [1042, 652], [1035, 613], [1028, 584], [1028, 551], [1028, 521], [1033, 491], [1046, 461], [1063, 432], [1073, 417], [1076, 403], [1083, 381], [1091, 367], [1098, 351], [1099, 339], [1092, 332], [1088, 315], [1087, 289], [1085, 273], [1085, 254], [1087, 236], [1102, 217], [1115, 199], [1135, 177], [1137, 169], [1140, 152], [1140, 139], [1137, 100], [1128, 73], [1122, 58], [1111, 43], [1088, 30], [1072, 24], [1048, 21], [1028, 14], [1005, 14], [983, 14], [966, 15], [942, 24], [925, 43], [907, 58], [895, 65], [884, 63], [881, 48], [877, 33], [865, 22], [831, 18], [811, 18], [788, 25]]
poly = []
for i in polygon:  poly.extend(i)
polygon_to_rle(poly)
import os
import json
from collections import defaultdict
import glob


def make_dir_dict(data_path):
    files = os.listdir(data_path)
    img2label = {f: 0 for f in files}
    label2img = {0: [[os.path.join(data_path, f), 0, 0, 100000, 100000] for f in files]}
    return img2label, label2img


def read_affectnet_dict(json_path, data_path='/home/misha/Documents/datasets/Affectnet/'):
    with open(json_path, 'r') as f:
        d = json.load(f)
    img2label = {}
    label2img = defaultdict(lambda: [])
    for k, values in d.items():
        if int(k) > 7:
            continue
        for v in values:
            # print(v)
            img_path = os.path.join(data_path, v[0].split('/Affectnet/')[1])
            img2label[img_path] = int(k)
            v[0] = img_path
            if v[1] != 'NULL':
                for i in range(1, 5):
                    v[i] = int(v[i])

            label2img[int(k)].append(v)
    return img2label, dict(label2img)

def add_rotations_affectnet(img2label, label2img, multiply_basic_ratio,
                            dirs=['/home/misha/Documents/datasets/Affectnet/data_rotated',
                                  '/home/misha/Documents/datasets/Affectnet/data_rotated_down']):
    for k in label2img.keys():
        label2img[k] = label2img[k] * multiply_basic_ratio
    for img_dir in dirs:
        dir_paths = glob.glob(img_dir + '/*/*')
        print('found {} images in {} directory'.format(len(dir_paths), img_dir))
        for p in dir_paths:
            p_dir, p_fn = os.path.split(p)
            p_fn, p_ext = os.path.splitext(p_fn)
            p_dir = p_dir.split('/')[-1]
            original_name = p_fn.split('_')[0]
            original_path = os.path.join('/home/misha/Documents/datasets/Affectnet/data/', p_dir, original_name + p_ext)
            if original_path in img2label:
                p_class = img2label[original_path]
                img2label[p] = p_class
                label2img[p_class].append([p, 0, 0, 1000, 1000])
                # print('added', p_class, p)
            # else:
            #     print(list(img2label.keys())[0], original_path)
            #     raise Exception
    return img2label, label2img




def read_face_count(json_path='/home/misha/Documents/datasets/Affectnet/train_faces_count.json'):
    with open(json_path, 'r') as f:
        d = json.load(f)
    return d
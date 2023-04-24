# -*- coding: utf-8 -*-
'''
@Time    : 6/7/2022 5:06 PM
@Author  : dong.yachao
'''

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse
import glob

from pathlib import Path
import glob
import random

classes = ['person']

abs_path = os.getcwd()


def convert(size, box):
    # box: xmin, ymin, xmax, ymax
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id,
                       xml_path='/project/train/src_repo/dataset/xmls/',
                       save_txt_dir_path='/project/train/src_repo/dataset/labels/'):
    # print('xml_dir_path:', xml_dir_path)
    in_file = open(xml_path, encoding='utf-8')
    out_file = open(save_txt_dir_path + image_id + '.txt', 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = "person"
        cls_id = classes.index(cls)
        # 获取bbox
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))

        # bbox转换 xyxy 转换为 xywh(0~1)
        bb = convert((w, h), b)
        # 写入 cls_id + color_id + bbox
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        # 换行
        out_file.write('\n')


def gen_imgs_path(label_txt_path='/home/data/vehicle_data/', train_ratio=0.9):
    '''
    根据 包含所有 all_det_imgs.txt 数据集绝对路径的 txt文件，分割成训练接、测试集、验证集
    '''
    label_path = os.path.dirname(label_txt_path)
    # 生成训练测试集，写入txt文件中
    with open(label_txt_path, 'r') as f1:
        all_det = f1.readlines()
        num_imgs = len(all_det)
        all_det = sorted(all_det)
        train_num = int(train_ratio * num_imgs)
        test_num = num_imgs - train_num
        # 按照固定间隔抽取训练集和测试集
        test_interval = int(num_imgs / test_num)
        test_abs_img_paths = all_det[::test_interval]
        train_abs_img_paths = list(set(all_det) - set(test_abs_img_paths))
        
        # 打乱随机取
        # num_imgs = len(all_det)
        # train_ratio = 0.9
        # train_abs_img_paths = all_det[:int(train_ratio * num_imgs)]
        # test_abs_img_paths = all_det[int(train_ratio * num_imgs):]

        with open(join(label_path, 'train.txt'), 'w') as f1:
            for train_pwd in train_abs_img_paths:
                f1.write(train_pwd)

        with open(join(label_path, 'test.txt'), 'w') as f1:
            for test_pwd in test_abs_img_paths:
                f1.write(test_pwd)

        with open(join(label_path, 'val.txt'), 'w') as f1:
            for val_pwd in test_abs_img_paths:
                f1.write(val_pwd)


if __name__ == '__main__':
    # 获取所有的img路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir_path', type=str, default='/home/data/vehicle_data/images/', help='img所在文件目录路径')

    parser.add_argument('--save_txt_dir_path', type=str, default='/home/data/person_data/labels/',
                        help='需要保存txt文件目录路径')

    parser.add_argument('--abs_img_txt_path', type=str, default='/home/data/person_data/all_imgs_path.txt',
                        help='存储img绝对路径的txt文件路径')
    opt = parser.parse_args()

    # 1. 将all_imgs_path.txt 分割为训练集测试集
    gen_imgs_path(label_txt_path=opt.abs_img_txt_path)
    # 2. xml 和 img 在同一目录文件夹下
    with open(opt.abs_img_txt_path, 'r') as f1:
        for xml in f1.readlines():
            image_name = xml.strip().split(os.sep)[-1].split('.')[0]
            convert_annotation(image_name, xml_path=xml.strip().replace('.jpg', '.xml'),
                               save_txt_dir_path=opt.save_txt_dir_path)


    # for personal win test
    # all_img_files_path = r'D:\CodeFiles\data\vehicle_data\vehicle\images'
    # all_imgs = os.listdir(all_img_files_path)
    # for image_name in all_imgs:
    #     abs_xml_path = os.path.join(all_img_files_path.replace('images', 'xmls'), image_name.replace('.jpg', '.xml'))
    #     convert_annotation(image_name.split('.')[0], xml_path=abs_xml_path, save_txt_dir_path=opt.save_txt_dir_path)


# -*- coding: utf-8 -*-
'''
@Time    : 4/18/2023 4:56 PM
@Author  : dong.yachao
'''
import os
from os.path import join
import xml.etree.ElementTree as ET



def gen_imgs_path(label_txt_path="/home/data/person_data/all_xml.txt", train_ratio=0.9):
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

        with open(join(label_path, 'train_xml.txt'), 'w') as f1:
            for train_pwd in train_abs_img_paths:
                f1.write(train_pwd)

        with open(join(label_path, 'test_xml.txt'), 'w') as f1:
            for test_pwd in test_abs_img_paths:
                f1.write(test_pwd)

        # with open(join(label_path, 'query.txt'), 'w') as f1:
        #     for val_pwd in test_abs_img_paths:
        #         f1.write(val_pwd)

if __name__ == '__main__':
    
    os.system('mkdir  -p /home/data/person_data/')
    os.system("find /home/data/*/ -name '*.xml' | xargs -i ls {}  > /home/data/person_data/all_xml.txt")
    
    gen_imgs_path(label_txt_path="/home/data/person_data/all_xml.txt", train_ratio=0.9)
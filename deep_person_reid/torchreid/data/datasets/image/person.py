from __future__ import division, print_function, absolute_import

import os.path
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class Person(ImageDataset):
    """
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """

    def __init__(self, root='', **kwargs):

        self.data_dir = root
        self.train_xml_path = osp.join(self.data_dir, 'train_xml.txt')
        # self.query_xml_path = osp.join(self.data_dir, 'query_xml.txt')
        self.test_xml_path = osp.join(self.data_dir, 'test_xml.txt')
        self.all_xml_path = osp.join(self.data_dir, 'all_xml.txt')

        self.pid2label, self.cid2label = self.get_all_id()

        train = self.process_dir(self.train_xml_path)
        test = self.process_dir(self.test_xml_path)

        num_test = len(test)
        test = sorted(test)
        query = test[::int(1/0.3)]
        gallery = list(set(test) - set(query))

        super(Person, self).__init__(train, query, gallery, **kwargs)

    def get_all_id(self):
        import xml.etree.ElementTree as ET
        # 遍历每一个xml文件
        pid_container = set()
        cid_container = set()
        with open(self.all_xml_path, "r") as f1:
            for one_xml in f1.readlines():
                one_xml = one_xml.strip()
                # 读取xml文件
                in_file = open(one_xml, encoding='utf-8')
                tree = ET.parse(in_file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    # 获取person id
                    person_id = int(str(obj.find('name').text).split("_")[-1])
                    pid_container.add(person_id)
                    # 获取cam id
                    cam_info = os.path.basename(one_xml).split("_")[:-1]
                    cam_scene = "".join(cam_info[:-1])
                    cam_id = cam_scene + "_" + cam_info[-1]
                    cid_container.add(cam_id)
                in_file.close()
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cid2label = {cid: label for label, cid in enumerate(cid_container)}
        return pid2label, cid2label

    def process_dir(self, xml_path):
        import xml.etree.ElementTree as ET
        # 遍历每一个xml文件
        data = []
        with open(xml_path, "r") as f1:
            for one_xml in f1.readlines():
                one_xml = one_xml.strip()
                # 读取xml文件
                in_file = open(one_xml, encoding='utf-8')
                tree = ET.parse(in_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                for obj in root.iter('object'):
                    # 获取该目标所在图像的路径
                    img_path = one_xml.replace(".xml", ".jpg")
                    # 获取person id
                    person_id = int(str(obj.find('name').text).split("_")[-1])
                    # relabel
                    person_id = self.pid2label[person_id]

                    # 获取cam id
                    cam_info = os.path.basename(one_xml).split("_")[:-1]
                    cam_scene = "".join(cam_info[:-1])
                    cam_id = cam_scene + "_" + cam_info[-1]
                    # relabel
                    try:
                        cam_id = self.cid2label[cam_id]
                    except:
                        print(f"img_path:{img_path}")

                    # 获取bbox
                    xmlbox = obj.find('bndbox')
                    box = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                           float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                    data.append(((img_path, box), person_id, cam_id))
                in_file.close()
        return data

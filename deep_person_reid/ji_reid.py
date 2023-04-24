# -*- coding: utf-8 -*-
'''
@Time    : 4/20/2023 10:10 AM
@Author  : dong.yachao
'''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from torchreid.utils import load_pretrained_weights, re_ranking
from torchreid.models import build_model
import torchvision.transforms as T
from torch.nn import functional as F
from PIL import Image
import torch
from torchreid import metrics
import numpy as np

class person_reid(object):
    def __init__(self, opt):
        self.opt = opt
        # Build transform functions
        transforms = []
        transforms += [T.Resize(self.opt.image_size)]
        transforms += [T.Resize((256, 128))]
        transforms += [T.ToTensor()]
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        if self.opt.pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        self.preprocess = T.Compose(transforms)
        self.to_pil = T.ToPILImage()
        self.reid_model = self.init_reid_model()
        self.gallery_info = None

    def init_reid_model(self):
        # Build model
        model = build_model(
            self.opt.reid_model_name,
            num_classes=1,
            pretrained=False,
            use_gpu=True
        )
        model.eval()
        load_pretrained_weights(model, self.opt.reid_model_path)
        model.to(self.opt.reid_device)
        device = torch.device(self.opt.reid_device)
        model.to(device)
        return model

    def extract_gallery_features(self, image_list):
        images = []
        pids = []
        cids = []
        for one_img in image_list:
            pid = one_img.split(os.sep)[-2]
            cid = one_img
            image = Image.open(one_img).convert('RGB')
            image = self.preprocess(image)
            images.append(image)

            pids.append(pid)
            cids.append(cid)
        images = torch.stack(images, dim=0)
        images = images.to(self.opt.reid_device)
        with torch.no_grad():
            features = self.reid_model(images)
        # features = features.cpu()
        if self.opt.normalize_feature:
            print('Normalzing features with L2 norm ...')
            features = F.normalize(features, p=2, dim=1)

        self.gallery_info = (features, pids, cids)
        # return (features, pids, cids)

    def extract_query_features(self, image):
        images = []
        # image = Image.open(image).convert('RGB')
        # image = self.to_pil(image)
        image = Image.fromarray(image).convert('RGB')
        
        image = self.preprocess(image)
        images.append(image)
        images = torch.stack(images, dim=0)
        images = images.to(self.opt.reid_device)
        with torch.no_grad():
            features = self.reid_model(images)
        # features = features.cpu()
        if self.opt.normalize_feature:
            print('Normalzing features with L2 norm ...')
            features = F.normalize(features, p=2, dim=1)
        # print("features.shape:", features.shape)
        return features

    def get_results(self, query_feats, gallery_info):
        gallery_feats, pids, cids = gallery_info
        distmat = metrics.compute_distance_matrix(query_feats, gallery_feats, self.opt.dist_metric)
        distmat = distmat.cpu().numpy()
        # rerank
        if self.opt.rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(query_feats, query_feats, self.opt.dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gallery_feats, gallery_feats, self.opt.dist_metric)
            distmat = re_ranking(distmat, distmat_qq.cpu().numpy(), distmat_gg.cpu().numpy())
        
        indices = np.argsort(distmat, axis=1)
        pred_id = pids[indices[0][0]]
        pred_cid = cids[indices[0][0]]
        return pred_id, pred_cid




'''
def init_reid_model(opt):
    # Build model
    model = build_model(
        opt.reid_model_name,
        num_classes=1,
        pretrained=False,
        use_gpu=True
    )
    model.eval()
    load_pretrained_weights(model, opt.reid_model_path)
    model.to(opt.reid_device)

    # Build transform functions
    transforms = []
    transforms += [T.Resize(opt.image_size)]
    transforms += [T.ToTensor()]

    pixel_mean = [0.485, 0.456, 0.406],
    pixel_std = [0.229, 0.224, 0.225]
    if opt.pixel_norm:
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    preprocess = T.Compose(transforms)
    # to_pil = T.ToPILImage()
    device = torch.device(opt.reid_device)
    model.to(device)

    return model, preprocess


def extract_gallery_features(image_list, model, preprocess, opt):
    images = []
    pids = []
    cids = []
    for one_img in image_list:
        pid = one_img.split()
        cid = one_img.split()
        image = Image.open(one_img).convert('RGB')
        image = preprocess(image)
        images.append(image)

        pids.append(pid)
        cids.append(cid)
    images = torch.stack(images, dim=0)
    images = images.to(opt.reid_device)
    with torch.no_grad():
        features = model(images)
    # features = features.cpu()
    if opt.normalize_feature:
        print('Normalzing features with L2 norm ...')
        features = F.normalize(features, p=2, dim=1)

    return (features, pids, cids)


def extract_query_features(image, model, preprocess, opt):
    to_pil = T.ToPILImage()
    images = []

    # image = Image.open(image).convert('RGB')
    image = to_pil(image)
    image = preprocess(image)
    images.append(image)
    images = torch.stack(images, dim=0)
    images = images.to(opt.reid_device)
    with torch.no_grad():
        features = model(images)
    # features = features.cpu()
    if opt.normalize_feature:
        print('Normalzing features with L2 norm ...')
        features = F.normalize(features, p=2, dim=1)

    return features


def get_results(query_feats, gallery_info, opt):
    gallery_feats, pids, cids = gallery_info

    distmat = metrics.compute_distance_matrix(query_feats, gallery_feats, opt.dist_metric)
    distmat = distmat.numpy()

    # rerank
    if opt.rerank:
        print('Applying person re-ranking ...')
        distmat_qq = metrics.compute_distance_matrix(query_feats, query_feats, opt.dist_metric)
        distmat_gg = metrics.compute_distance_matrix(gallery_feats, gallery_feats, opt.dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    indices = np.argsort(distmat, axis=1)
    pred_id = pids[indices[0][0]]
    pred_cid = cids[indices[0][0]]
    return pred_id, pred_cid
'''
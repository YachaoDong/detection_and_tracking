import json
import argparse
import os
import sys
import torch
import cv2

# python ji.py --weights '/project/train/src_repo/detection_and_tracking/yolov7/yolov7_training.pt' --gallery_dir  /home/data/
sys.path.append('/project/train/src_repo/detection_and_tracking/')
# YOLO Det
sys.path.append('/project/train/src_repo/detection_and_tracking/yolov7')  # add ROOT to PATH
from yolov7.ji_detect import person_detection
# reid model api
# sys.path.append('/project/train/src_repo/detection_and_tracking/deep_person_reid/')
from deep_person_reid.ji_reid import person_reid

# 参数
# /project/train/models/exp3/weights/best.pt
parser = argparse.ArgumentParser()
# person detection args
parser.add_argument('--weights', nargs='+', type=str, default='/project/train/models/exp2/weights/best.pt',
                    help='/project/train/src_repo/detection_and_tracking/yolov7/yolov7_training.pt')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default="0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# reid args
parser.add_argument('--reid_model_name', type=str, default='osnet_ain_x1_0', help='reid model.pt path(s)')
parser.add_argument('--reid_model_path', type=str, default='/project/train/models/reid/model/model.pth.tar-2',
                    help='/project/train/models/reid/model/model.pth.tar-2')
parser.add_argument('--gallery_dir', type=str, default="/home/person_imgs/", help="/home/person_imgs/")
parser.add_argument('--reid_device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--image_size', type=tuple, default=(256, 128), help='inference size (pixels)')
parser.add_argument('--pixel_norm', type=bool, default=True, help='object confidence threshold')
parser.add_argument('--normalize_feature', type=bool, default=False, help='object confidence threshold')
parser.add_argument('--dist_metric', type=str, default='cosine', help='cosine or euclidean')
parser.add_argument('--rerank', type=bool, default=True, help='object confidence threshold')

opt = parser.parse_args()
print(opt)
opt = parser.parse_args()


@torch.no_grad()
def init():
    # init person detection model
    dect_instance = person_detection(opt)

    # init reid model
    reid_instance = person_reid(opt)
    # get gallery features
    all_image_list = []
    for root, dirs, files in os.walk(opt.gallery_dir):
        for file in files:
            if file.endswith(".jpg"):
                all_image_list.append(os.path.join(root, file))
    # print("all_image_list:", all_image_list)
    reid_instance.extract_gallery_features(all_image_list)
    return [dect_instance, reid_instance]


@torch.no_grad()
def process_image(handle=None, input_image=None, args=None, **kwargs):
    '''Do inference to analysis input_image and get output
    Attributes:
    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    '''
    args = json.loads(args)
    cid = args['cid']
    fid = args['fid']

    fake_result = {}
    # fake_result['algorithm_data'] = {}
    # fake_result['algorithm_data']["target_info"] = []

    fake_result['model_data'] = {}
    fake_result['model_data']["objects"] = []

    dect_instance = handle[0]
    reid_instance = handle[1]

    # detection
    det_pred = dect_instance.detect(input_image)
    for i, (cropped_person, xyxy, conf, cls_name) in enumerate(det_pred):
        # reid
        query_features = reid_instance.extract_query_features(cropped_person)
        pred_id, cam_cid = reid_instance.get_results(query_features, reid_instance.gallery_info)
        fake_result['model_data']['objects'].append(
            {
                "cid": cid,
                "fid": fid,
                "x": int(xyxy[0]),
                "y": int(xyxy[1]),
                "height": int(xyxy[3] - xyxy[1]),
                "width": int(xyxy[2] - xyxy[0]),
                "name": str(cls_name),
                "confidence": float(conf),
                "id": int(pred_id)
            })
    # print("fake_result:", fake_result)
    return json.dumps(fake_result, indent=4)


if __name__ == '__main__':
    handle = init()
    # person_10005
    img = cv2.imread('/home/data/2130/office_17_000520.jpg')
    img = img[303:734, 456:670]
    process_image(handle=handle, input_image=img, args='{"fid":"0", "cid":"0"}')

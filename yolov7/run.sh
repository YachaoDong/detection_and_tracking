#run.sh
# bash /project/train/src_repo/detection_and_tracking/yolov7/run.sh

# 如果存在要保存的文件，提前删除文件夹
rm  -r /home/data/person_data/*


#创建数据集相关文件夹
# mkdir  -p /project/.config/Ultralytics/
mkdir  -p /home/data/person_data/labels

# 生成all .jpg  abs path txt
find /home/data/*/ -name "*.jpg" | xargs -i ls {}  > /home/data/person_data/all_imgs_path.txt

# xml转txt labels
python /project/train/src_repo/detection_and_tracking/yolov7/xml2labels.py



# 开发环境训练demo 调试
# python /project/train/src_repo/detection_and_tracking/yolov7/train.py   --batch-size 4 --weights /project/train/src_repo/detection_and_tracking/yolov7/yolov7_training.pt --epochs 1 --workers 4


# 正式训练 pretrained model
# python /project/train/src_repo/detection_and_tracking/yolov7/train.py   --batch-size 4 --weights /project/train/src_repo/detection_and_tracking/yolov7/yolov7_training.pt --epochs 200 --workers 4


# 正式训练
# python /project/train/src_repo/detection_and_tracking/yolov7/train.py   --batch-size 4 --weights /project/train/models/exp/weights/last.pt --epochs 120 --workers 4


# resume训练
python /project/train/src_repo/detection_and_tracking/yolov7/train.py   --batch-size 4  --resume  /project/train/models/exp2/weights/last.pt --epochs 120 --workers 4



# python /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/train.py   --batch-size 16 --weights /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/yolov7_training.pt  --data /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/data/vehicle.yaml  --hyp /project/train/src_repo/yolov7_vehicle_plate_det/yolov7-main/data/hyp.scratch.vehicle_custom.yaml --epochs 100 --workers 4 --multi-scale


# 测试
# python /project/train/src_repo/yolov5_vehicle_plate_det/yolov5_det/val.py --data /project/train/src_repo/yolov5_vehicle_plate_det/yolov5_det/data/vehicle.yaml --weights /project/train/models/exp2/weights/last.pt




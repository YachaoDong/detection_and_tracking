# bash /project/train/src_repo/detection_and_tracking/deep-person-reid/reid_run.sh 

# 1.生成标签
python /project/train/src_repo/detection_and_tracking/deep-person-reid/gen_xml.py

# 2.训练
cd /project/train/src_repo/detection_and_tracking/deep-person-reid/
python /project/train/src_repo/detection_and_tracking/deep-person-reid/scripts/main.py
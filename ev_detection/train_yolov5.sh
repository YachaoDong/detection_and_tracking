CURRENT_DIR=$(cd $(dirname $0); pwd)
echo "${CURRENT_DIR}"
cd "${CURRENT_DIR}"
export YOLOV5_CONFIG_DIR="${CURRENT_DIR}/configs"

#把数据分为训练和测试, 可以修改相应比例
python splitData.py
# 需要修改data/EVDATA.yaml的类别, modelYaml/yolov5.yaml中nc的数量
python train_det.py --mode yolov5 --data data/EVDATA.yaml --exist_ok --cfg modelYaml/yolov5s.yaml --epochs 50 --hyp data/hyps/hyp.scratch.yaml --weights yolov5s.pt --batch-size 8 --project /project/train/models
# 导出export
python tools/export.py --mode yolov5 --weights /project/train/models/exp/weights/best.pt --img 640
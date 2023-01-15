import os

from mmdet.apis import init_detector, inference_detector


def demo_mmdet():
    base_dir = r'C:\Pycharm_Projects\mmdetection-master'       # mmdetection的安装目录

    config_file = os.path.join(base_dir, r'configs/yolact/yolact_rfp.py')
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = r'work_dirs/yolact_rfp_data1384×5/latest.pth'

    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 测试单张图片并展示结果
    img = os.path.join(base_dir, r'demo\t4.png') # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
    result = inference_detector(model, img)
    # 在一个新的窗口中将结果可视化
    model.show_result(img, result, out_file=None, show=True)


if __name__ == '__main__':
    demo_mmdet()


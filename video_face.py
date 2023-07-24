import os
import cv2
from os.path import join
from face_detector_s3fd.detect_engine import FaceDetectionEngine

detector = FaceDetectionEngine(weights_path="face_detector/s3fd_convert.pth")


def video(input_path, output_path):
    for img_name in os.listdir(input_path):
        img = cv2.imread(join(input_path, img_name))  # 读取图片
        preds = detector.predict(img, dilate_bbox=True)  # 人脸检测
        crops = [detector.crop(img, bbox) for bbox in preds]  # 人脸切割
        try:
            cv2.imwrite(join(output_path, img_name), crops[0])  # 进行保存
        except Exception:
            pass


data_path = './video1'
for video_name in os.listdir(data_path):  # 循环处理视频文件夹
    os.makedirs(join('./video1_img', video_name), exist_ok=True)  # 创建文件夹
    video(join(data_path, video_name), join('./video1_img', video_name))  # 对单个视频文件及进行处理

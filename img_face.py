import cv2
import os
from face_detector_s3fd.detect_engine import FaceDetectionEngine

detector = FaceDetectionEngine(weights_path="face_detector/s3fd_convert.pth")

img = cv2.imread("multiple_faces.jpg")  # 读取单张图片
preds = detector.predict(img, dilate_bbox=True)  # 人脸检测
crops = [detector.crop(img, bbox) for bbox in preds]  # 人脸切割

os.makedirs('save', exist_ok=True)
for i in range(len(crops)):  # 循环保存人脸
    cv2.imwrite(f'save/{str(i)}.png', crops[i])

# 画人脸框
for index, bbox in enumerate(preds):
    if bbox is not None:
        ymin, xmin, ymax, xmax = [int(i) for i in bbox]
        cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (0, 0, 255), 2)  # 画图
cv2.imwrite('mul_rec.png', img)  # 保存

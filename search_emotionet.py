import os
import glob

import cv2

from face_det_python import scrfd
from scn_python import scn

# face detection handle
fd = scrfd.ScrdfFaceDet(0.45)

# facial expression handle
fer = scn.ScnFacialExpressionCat()

image_dir = '/Users/zhourui/workspace/pro/facialExpression/data/org/emotioNet/dataFile_1094'
imgs = glob.glob(image_dir + '/*.jpg')

for img in imgs:
    # imread
    image = cv2.imread(img)

    # face detction
    result = fd.forward(image)
    if len(result) > 1:
        continue
    box = result[0]
    sx,sy,ex,ey,prob = box

    # facial expression
    label, scores = fer(image, [sx,sy,ex,ey])
    print(scores)


    # for debug
    # cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 2)
    # cv2.putText(image, '{}'.format(fer.facial_label_to_name(label)), (sx, sy-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # for box in result:
    #     rect = box[:4]
    #     prob = box[4]
    #     cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)


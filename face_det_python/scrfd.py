import sys
import os

from mmdet.apis import inference_detector, init_detector
import cv2

class ScrdfFaceDet(object):
    def __init__(self, score_thr, model_path='models/model.pth', device='cpu', config='models/scrfd_500m.py'):
        """
        SCRFD face detector
        """
        self.score_thr = score_thr
        self.model = init_detector(config, model_path, device=device)

    def forward(self, image):
        """
        forward with image path or numpy array
        :param image_path:
        :return: [[sx,sy,ex,ey,prob], [...]]
        """
        result = inference_detector(self.model, image)[0]
        ret = []
        for box in result:
            rect = [int(x) for x in box[:4]]
            prob = box[4]
            if prob<self.score_thr:
                break
            rect.append(prob)
            ret.append(rect)
        return ret

def main():
    import glob

    fd = ScrdfFaceDet(0.3)

    image_dir = '/Users/zhourui/workspace/pro/facialExpression/data/org/emotioNet/dataFile_1094'
    imgs = glob.glob(image_dir+'/*.jpg')

    for img in imgs:
        image = cv2.imread(img)
        result = fd.forward(image)
        print(result)
        for box in result:
            rect = box[:4]
            prob = box[4]
            cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), (255,0,0), 1)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        # show the results
        #show_result_pyplot(model, img, result, score_thr=score_thr)


if __name__ == '__main__':
    main()

import sys
import os
from mmcls.apis import inference_model, init_model, show_result_pyplot
import cv2

dir_label_map = {0:'Angry', 1:'Happy', 2:'Neutral', 3:'Sad', 4:'Surprise'}

class MMCLSFer(object):
    def __init__(self, config_file_path='models/mobilenet_v2/mobilenet_v2.py', ckpt_path='models/mobilenet_v2/latest.pth', device='cpu', input_channels=3):
        """
        mmcls facial expression
        """
        self.input_channels = input_channels
        self.model = init_model(config_file_path, ckpt_path, device=device)
        self.model.CLASSES = list(dir_label_map)

    def get_input_face(self, image, rect):
        sx, sy, ex, ey = rect
        h, w, c = image.shape
        faceh = ey - sy
        facew = ex - sx

        longsize = max(faceh, facew)
        expendw = longsize - facew
        expendh = longsize - faceh

        sx = sx - (expendw / 2)
        ex = ex + (expendw / 2)
        sy = sy - (expendh / 2)
        ey = ey + (expendh / 2)

        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(w - 1, ex))
        ey = int(min(h - 1, ey))

        return image[sy:ey, sx:ex, :], sx, sy, ex, ey

    def __call__(self, image, face_rect):
        """
        forward with image path or numpy array
        :param image_path:
        :return: [[sx,sy,ex,ey,prob], [...]]
        """
        image_face, isx, isy, iex, iey = self.get_input_face(image, face_rect)

        if self.input_channels==1:
            image_face = cv2.cvtColor(image_face, cv2.COLOR_BGR2GRAY)

        # inference image with
        result = inference_model(self.model, image_face, face_rect=None)

        return result['pred_label'], result['pred_score'], dir_label_map[result['pred_label']]

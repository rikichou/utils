import cv2
import torch
from torchvision import transforms
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import time

preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes=7):
        super(Res18Feature, self).__init__()
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(
            *list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children(
        ))[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out

class ScnFacialExpressionCat(object):
    """
    Facial expression recognition use SCN
    """
    def __init__(self, model_path='scn_python/models/epoch26_acc0.8615.pth', device='cpu'):
        self.device = torch.device("cuda:0" if device=='cuda' else "cpu")
        self.model = Res18Feature(pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.m = torch.nn.Softmax(dim=1)
    
    def __str__(self):
        return '0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral'

    def facial_label_to_name(self, label):
        label_map = {0:'Surprise', 1:'Fear', 2:'Disgust', 3:'Happiness', 4:'Sadness', 5:'Anger', 6:'Neutral'}
        return label_map[label]

    def forward(self, image):
        """
            image: color BGR

            return:
                pred_label: 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
                scores: predict prob
        """
        image = image[:, :, ::-1]  # BGR to RGB
        image_tensor = preprocess_transform(image)
        tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(),
                        requires_grad=False)
        tensor = tensor.to(self.device)
        _, outputs = self.model(tensor)

        pred_label = torch.argmax(outputs, 1).item()
        scores = self.m(outputs).cpu().detach().squeeze().numpy()

        return pred_label, scores[int(pred_label)], self.facial_label_to_name(int(pred_label))

    def get_input_face(self, image, rect):
        sx,sy,ex,ey = rect
        h,w,c = image.shape
        faceh = ey-sy
        facew = ex-sx

        longsize = max(faceh, facew)
        expendw = longsize-facew
        expendh  = longsize-faceh

        sx = sx-(expendw/2)
        ex = ex+(expendw/2)
        sy = sy-(expendh/2)
        ey = ey+(expendh/2)

        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(w-1, ex))
        ey = int(min(h-1, ey))

        return image[sy:ey, sx:ex, :]

    def __call__(self, image, rect):
        """
        facial expression recog
        :param image:   bgr format, 3 channel
        :param rect:    [sx, sy, ex, ey]
        :return:    pred_label, scores
        """
        image = self.get_input_face(image, rect)
        return self.forward(image)

import sys
sys.path.insert(0,'.')
sys.path.insert(0,'..')
import torch
import os
import cv2
import tensorflow as tf
from pytorch2keras.converter import pytorch_to_keras
from tensorflow.keras.models import Model
from models import FeatherNet
from models import mobilenetv2 
from models import MobileLiteNet
from torchsummary import summary
from PIL import Image
import torchvision.transforms as transforms
import time
import copy
import mediapipe as mp
import numpy as np


def draw_detection(
    image,
    bboxes,
    as_results,
):  
   
    for bbox in bboxes:
     
        # as_results
        # 0: real - 1: fake
        if  as_results== 0:
            cv2.rectangle(image, (bbox[0], bbox[2]), (bbox[1], bbox[3]),
                         (255, 0, 0), 2)

            cv2.putText(image, str("real"), (bbox[0], bbox[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)

        elif as_results == 1:
            cv2.rectangle(image, (bbox[0], bbox[2]), (bbox[1], bbox[3]),
                         (0, 0, 255), 2)
            
            cv2.putText(image, str("fake"),
                       (bbox[0], bbox[2] - 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 1,  cv2.LINE_AA)    

    return image


def run_face_detection(
        face_detection,
        image,
        expansion_rate=[0.1, 0.4, 0.1, 0.0],  # x1, y1, x2, y2
):
    input_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    results = face_detection.process(input_image)

    image_width, image_height = image.shape[1], image.shape[0]
    bboxes = []
    scores = []

    if results.detections is not None:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * image_width)
            y1 = int(bbox.ymin * image_height) 
            w = int(bbox.width * image_width)
            h = int(bbox.height * image_height)

            x1 = x1 - int(w * expansion_rate[0])
            y1 = y1 - int(h * expansion_rate[1])
            x2 = x1 + w + int(w * expansion_rate[0]) + int(w * expansion_rate[2])
            y2 = y1 + w + int(h * expansion_rate[1]) + int(h * expansion_rate[3])

            x1=np.clip(x1, 0 , image_width)
            x2=np.clip(x2, 0, image_width)
            y1=np.clip(y1, 0, image_height)
            y2=np.clip(y2, 0, image_height)

            bboxes.append([x1, x2, y1, y2])

            scores.append(detection.score[0])
    
    return bboxes, scores

def check_spoofing(image):
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil=Image.fromarray(img)
    image1 = transform(img_pil).unsqueeze(0)
    output1 = model(image1)
    #print(output1)
    soft_output = torch.softmax(output1, dim=-1)
    #print(soft_output,soft_output.data)
    preds = soft_output.to('cpu').detach().numpy()
    #print(preds)
    _, predicted = torch.max(soft_output.data, 1)
    predicted = predicted.to('cpu').detach().numpy()

    return predicted[0]

if __name__=='__main__':
    input_size=224
    img_size=224
    ratio = 224.0 / float(img_size)
    global normalize, transform
     # Data loading code
    normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],  ##accorcoding to casia-surf val to commpute
                                     std=[0.10050353, 0.100842826, 0.10034215])
    name = 'FeatherNetB'
    model = FeatherNet(se = True,avgdown=True)
    
    # model = MobileLiteNet54_se()
    # name='Mobilenetv2'
    # model = mobilenetv2.moilenetv2()

    #print(model)
    model_path = '/home/fuonq/Code/anti-spoof/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019/our_pretrained_models/FeatherNetB_bs32/_47_best.pth.tar'
    checkpoint = torch.load(model_path,map_location = 'cpu')
    # print('load model:',model_path)
    model_dict = {}
    state_dict = model.state_dict()
    #print(checkpoint)
    for (k,v) in checkpoint['state_dict'].items():
        print(k)
        if k[7:] in state_dict:
            model_dict[k[7:]] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    #model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    fd_model_selection = 0
    min_detection_confidence = 0.6

    # face-detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=fd_model_selection,
        min_detection_confidence=min_detection_confidence,
    )

    cap_device = 0
    cap_width = 1280
    cap_height = 720

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    while True:
      
        ret, image = cap.read()
        if not ret:
            break

        image_cp=copy.deepcopy(image)

        bbox, socres=run_face_detection(face_detection, image)
        # predict_anti=[]
      
        predict_anti=check_spoofing(image)
        
        if not len(bbox):
            continue

        image_cp=draw_detection(image_cp, bbox, predict_anti)

        key=cv2.waitKey(1)
        if key==27:
            break

        cv2.imshow("Anti-spoofing", image_cp)

    cap.release()
    cap.destroyAllWindows()



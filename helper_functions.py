import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import xml.etree.ElementTree as ET
import numpy as np
import PIL.Image as Image
from sklearn.datasets import load_files

def read_pascal(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(float(box.find("ymin").text))
            xmin = int(float(box.find("xmin").text))
            ymax = int(float(box.find("ymax").text))
            xmax = int(float(box.find("xmax").text))

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return filename, list_with_all_boxes

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
  
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
   
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
   
    iou = interArea / float(boxAArea + boxBArea - interArea)
   
    return iou


def load_data():
    data = load_files('use_data')
    filenames = data['filenames']
    targets = data['target']
    target_names = data['target_names']

    X = []
    y = []
  
    not_count = 0
    for name in filenames:
        if 'not' in name:
            not_count+=1
            if not_count < 2300:
                img = cv2.imread(name)
                X.append(img)
                y.append(0)
            else:
                pass
        else:
            img = cv2.imread(name)
            X.append(img)
            y.append(1)

    return np.array(X),y



def non_max_suppression_fast(boxes, overlapThresh,probs):
   
    if len(boxes) == 0:
        return []

    
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    final_boxes = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
        np.where(overlap > overlapThresh)[0])))

        
        final_boxes.append(boxes[pick])
      
    return boxes[pick]

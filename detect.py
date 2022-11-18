import numpy as np

import utils.plattSMO
import torch
import cv2
import yaml
from utils import Roiselect,svm_image
from model import SegNet

class Restore():
    def __init__(self,anchor,img,class_num = 2):
        self.img = img
        if len(img.shape)==3:
            self.h,self.w,self.c = img.shape
        else:
            self.h, self.w = img.shape
        self.anchor = anchor
        self.res = np.zeros([self.h,self.w,class_num])
        self.keylist = []
        for k,v in anchor.items():
            self.keylist.append(k)
        self.iter = 0

    def __getitem__(self, item):
        box = np.array(self.anchor[item])
        img = self.img[box[0,1]:box[1,1],box[0,0]:box[1,0],:]
        img = cv2.resize(img,[self.w,self.h])
        return img

    def __setitem__(self, key, values):
        box = np.array(self.anchor[key])
        value,c = values
        value = cv2.resize(np.float_(value),[box[1, 0] - box[0, 0],box[1, 1] - box[0, 1]])
        res = np.zeros_like(self.res)
        res[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0], c] = value
        self.res += res

    def __str__(self):
        return str(self.res)

    def __next__(self):
        item = self.keylist[self.iter]
        self.iter += 1
        box = self.anchor[item]
        img = self.img[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0], :]
        img = cv2.resize(img, [self.w, self.h])
        return img
    def __iter__(self):
        return self

    def getdet(self):
        return np.where(self.res>0,1,0)

def detect(net,img,config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    svm_res = svm_image(img,'YBCr')
    cal_area = lambda x: ((x[1][0] - x[0][0]) * (x[1][1] - x[0][1]))
    con = Roiselect(svm_res)
    con.connect_(200)
    anchor = con.getanchor()
    size=config['size_img']
    img = cv2.resize(img,[size[1],size[0]])
    detect_machine = Restore(anchor,img)
    for k,v in anchor.items():
        if cal_area(v) < 40:
            continue
        img_ko = detect_machine[k]
        img_k = img_ko.transpose([2,0,1])
        img_i = torch.from_numpy(img_k).unsqueeze(0).to(torch.float).to(device)
        mask,y = net(img_i)
        mask = torch.argmax(mask[0].detach(),0)
        mask_numpy = mask.to(torch.device('cpu')).numpy()
        class_ = int(torch.argmax(y,1))
        detect_machine[k] = mask_numpy,class_
    return detect_machine.getdet()

def main():
    img = cv2.imread(r'D:\dataset_sweet\image\_2016-05-27-10-26-48_5_frame1.png')
    net = SegNet(2)
    with open(r'config/config.yaml','r') as f:
        config = yaml.load(f,yaml.FullLoader)
    t = torch.load(config['weight_path'])
    net.load_state_dict(t)
    c_res = detect(net,img,config)
    size = config['size_img']
    size.append(3)
    img_res = np.zeros(size)
    img_res[:,:,1:] = c_res
    cv2.imshow('detect result',img_res)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
from networks import FaceBox
from encoderl import DataEncoder
from numpy import *
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from tqdm import tqdm
print('opencv version', cv2.__version__)

use_gpu = False



def detect(im):
    #图像缩放
    im = cv2.resize(im, (1024,1024))

    #torch.from_numpy(ndarry)，从numpy中获得数据     numpy--->tensor
    #生成返回的tensor会和ndarry共享数据，任何对tensor的操作都会影响到ndarry,反之亦然
    im_tensor = torch.from_numpy(im.transpose((2,0,1)))#transpose转置
    im_tensor = im_tensor.float().div(255)  #除以255，使数据转变到 [0,1] 之间
    #print(im_tensor.shape)
    #torch.unsqueeze()这个函数主要是对数据维度进行扩充
    #b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度
    #locatization位置   confidence 置信度
    loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0), volatile=True))
    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0),
                                                F.softmax(conf.squeeze(0)).data)
    return boxes, probs

def detect_gpu(im):
    im = cv2.resize(im, (1024,1024))
    im_tensor = torch.from_numpy(im.transpose((2,0,1)))
    im_tensor = im_tensor.float().div(255)
    # print(im_tensor.shape)
    loc, conf = net(Variable(torch.unsqueeze(im_tensor, 0), volatile=True).cuda())
    loc, conf = loc.cpu(), conf.cpu()
    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0),
                                                F.softmax(conf.squeeze(0)).data)
    return boxes, probs


def testIm(file,picture):
    #加载图片，返回一个[height, width, channel]的numpy.ndarray对象
    # height表示图片高度，width表示图片宽度，channel表示图片的通道。
    im = cv2.imread(file)
    if im is None:
        print("can not open image:", file)
        return
    #im.shape获得图像的形状，返回值是一个包含行数，列数，通道数的元组
    h,w,_ = im.shape
    boxes, probs = detect(im)
    print(boxes)
    num=0
    for i, (box) in enumerate(boxes):
        num=num+1
        print('i', i, 'box', box)
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        print(x1, y1, x2, y2, w, h)
        cv2.rectangle(im,(x1,y1+4),(x2,y2),(0,0,255),2)
        cv2.putText(im, str(np.round(probs[i],2)), (x1,y1), font, 0.4, (0,0,255))
    #cv2.imwrite('photo2.jpg', im)
    cv2.imwrite('picture/' + picture, im)

    src = cv2.imread('picture/' + picture)
    print("图中人数为:",num)
    text="number of people: "+str(num)
    cv2.putText(src, text, (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 5)
    cv2.imshow('text', src)
    cv2.waitKey()
    return im

if __name__ == '__main__':
    net = FaceBox()
    #加载模型
    net.load_state_dict(torch.load('weight/faceboxes.pt', map_location=lambda storage, loc:storage))
    
    if use_gpu:
        net.cuda()
    #调用模型
    net.eval()
    data_encoder = DataEncoder()#?

    #设置opencv字体
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

    # given image path, predict and show
    imgpath = "./img/"
    picture = '005.jpg'
    im = testIm(imgpath + picture,picture)
   # im = testIml(fddb_path ,picture)
   # cv2.imwrite('picture/'+picture, im)



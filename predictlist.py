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

def testIm(file):
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

    src = cv2.imread('photo2.jpg')
    print("图中人数为:",num)
    text="number of people: "+str(num)
    cv2.putText(src, text, (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 5)
    cv2.imshow('text', src)
    cv2.waitKey()
    return im

#imgpath='widerface/WIDER_val/images/'
def testIml(imgpath,imgname):
    #加载图片，返回一个[height, width, channel]的numpy.ndarray对象
    # height表示图片高度，width表示图片宽度，channel表示图片的通道。

    file=imgpath+imgname

    im = cv2.imread(file)
    if im is None:
        print("can not open image:", file)
        return
    #im.shape获得图像的形状，返回值是一个包含行数，列数，通道数的元组
    h,w,_ = im.shape

    try:
        boxes, probs = detect(im)
    except:
        return im
    #print(boxes)
    num=0
    for i, (box) in enumerate(boxes):
        num=num+1
    print('检测到'+imgname+"中有"+str(num)+"张人脸")


    #outputs.writelines('检测到' + imgname + "中有" + str(num) + "张人脸"+'\n')
    #outputs.writelines("每张人脸位置如下"+'\n')

    #print("每张人脸位置如下")
    for i, (box) in enumerate(boxes):
        #print('i', i, 'box', box)
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        t1 = round(float(x1), 4)
        t2 = round(float(y1), 4)
        t3 = round(float(x2-x1), 4)
        t4 = round(float(y2-y1), 4)
       # outputs.writelines(str(t1)+' '+str(t2)+' '+str(t3)+' '+str(t4)+'\n')
        #print(t1,t2,t3,t4)

        #cv2.rectangle(im,(x1,y1+4),(x2,y2),(0,0,255),2)
   # cv2.imwrite('widerface/WIDER_val_result/'+imgname, im)
    #cv2.imwrite('./1/1.jpg', im)

    truenum=findnum(imgname)  #人脸数量真值

    if(truenum==-1 or truenum<num):
        truenum=num
    print(imgname + '人脸数量真值为'+str(truenum))

    #outputs.writelines(imgname + '人脸数量真值为'+str(truenum)+'\n')

    rate=num/truenum  #识别率
    print("识别率为"+str(rate))

   # outputs.writelines("识别率为"+str(rate)+'\n')
    #outputs.writelines('---------------------------------------------------------------------'+'\n')


    print("--------------------------------------------------------")


    if truenum<=15:
        rate0_15.append(rate)
    if truenum>=16 and truenum<=30:
        rate16_30.append(rate)
    if truenum>=31 and truenum<=50:
        rate31_50.append(rate)
    if truenum>=51 and truenum<=100:
        rate51_100.append(rate)
    if truenum>=101 and truenum<=150:
        rate101_150.append(rate)
    if truenum>150:
        rate151_.append(rate)






    return im

#找人脸数量真值
def findnum(imgname):
    path='widerface/wider_face_split/'
    flag=False  #标记，是否找到对应行
    with open(path+'wider_face_val_bbx_gt.txt') as f:
        lines = f.readlines()
        length = len(lines)
    for i in range(length):

        line = lines[i]

        if flag==True:
            #print(imgname+'人脸真值为'+line)
            #防止取到的不是数字
            try:
                return int(line)
            except:
                return -1
        if imgname  in line:
            flag=True
    return -1  #没找到




def testImList(path, file_name):
    with open(path+file_name) as f:
        file_list = f.readlines()

    for item in file_list:
        imgpath='widerface/WIDER_val/images/'
        #testIm(path+item.strip()+'.jpg')
        testIml(imgpath,item.strip())
        #print(imgpath+item.strip())

def saveFddbData(path, file_name):
    '''
    Args:
        file_name: fddb image list
    '''
    with open(path+file_name) as f:
        file_list = f.readlines()
    f_write = open('predict.txt', 'w')
    
    image_num = 0
    for item in tqdm(file_list):
        item = item.strip()
        if not ('/' in item):
            continue
        image_num += 1
        im = cv2.imread(path+item+'.jpg')
        if im is None:
            print('can not open image', item)
            return
        h,w,_ = im.shape
        if use_gpu:
            boxes, probs = detect_gpu(im)
        else:
            boxes, probs = detect(im)
        f_write.write(item+'\n')
        f_write.write(str(boxes.size(0))+'\n')
        # print('image_num', image_num, 'box_num', boxes.size(0))
        for i, (box) in enumerate(boxes):
            x1 = box[0]*w
            x2 = box[2]*w
            y1 = box[1]*h
            y2 = box[3]*h
            t1=round(float(x1),4)
            t2=round(float(y1),4)
            t3=round(float(x2-x1),4)
            t4=round(float(y2-y1),4)
            prob=round(float(probs[i]),10)
            #print(type(x1));
            #print(x1,x2,y1,y2,prob);
            #f_write.write(str(x1)+'\t'+str(y1)+'\t'+str(x2-x1)+'\t'+str(y2-y1)+'\t'+str(prob)+'\t'+'1\n')
            f_write.write(str(t1)+' '+str(t2)+' '+str(t3)+' '+str(t4)+' '+str(prob)+' '+'\n')
    f_write.close()

def getFddbList(path, file_name):
    with open(path+file_name) as f:
        file_list = f.readlines()
    f_write = open(path+'fddblist.txt', 'w')
    for item in file_list:
        if '/' in item:
            f_write.write(item)
    f_write.close()
    print('get fddb list done')


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





    # 写入txt文件
    #txtpath = 'D:/file/pyproject/faceboxes/widerface/wider_face_split/'
    #outputs = open(txtpath + 'result.txt', 'w')

    # 统计分布

    rate0_15 = []
    rate16_30 = []
    rate31_50 = []
    rate51_100 = []
    rate101_150 = []
    rate151_ = []

    path='widerface/wider_face_split/'
    file_name ='val_list.txt'
    testImList(path, file_name)

    ave=rate0_15+rate16_30+rate31_50+rate51_100+rate101_150+rate151_

    print("不同人脸数量区间下的平均识别率：")
    print("0-15："+str(mean(rate0_15)))
    print("16-30："+str(mean(rate16_30)))
    print("31-50："+str(mean(rate31_50)))
    print("51-100："+str(mean(rate51_100)))
    print("101-150："+str(mean(rate101_150)))
    print("151+"+str(mean(rate151_)))

    print("总识率别为："+str(mean(ave)))


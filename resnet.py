import time
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from PIL import Image

# -------------------------------------------------------------------------------------------------------
# Function Name：    model_init
# Function Usage：   resnet模型的初始化
# paras [in]：       model_name        使用的模型名称
# paras [out]：      model             初始化的模型
#                   transform_test    预测参数
# -------------------------------------------------------------------------------------------------------
def model_init(model_name):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_name)
    model.eval()
    model.to(DEVICE)
    return model, transform_test

# -------------------------------------------------------------------------------------------------------
# Function Name：    resnet_predict
# Function Usage：   利用模型来预测当前输入图像隶属于数据库中的位置
# paras [in]：       transform_test    模型参数
#                    model            使用的模型
#                    img              要输入的图片（用Img.open函数打开的图像）
# paras [out]：      predictclass     预测出来的模型的类名
#                    classid          预测出来的模型的类号
#                    duration         预测所用的时间
# -------------------------------------------------------------------------------------------------------
def resnet_predict(transform_test, model, img):
    # 这里的img是用Img.open函数打开的图像

    # 预存会用到的类
    classes = ('CNYH', 'CXX', 'DIANLOUTIEIG', 'DIANTISEV', 'FUTIEIG',
               'FUTISEV', 'HJH', 'LC', 'LOUTISEV', 'LY', 'NRQC',
               'SMHY', 'WL', 'XCY', 'YJAL', 'YK', 'YZJ', 'ZDJ')

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.process_time()
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)

    _, pred = torch.max(out.data, 1)
    print('当前位置:{}'.format(classes[pred.data.item()]))
    classid = pred.data.item()
    predictclass = classes[pred.data.item()]
    end_time = time.process_time()
    duration = end_time - start_time

    return predictclass, classid, duration

if __name__ =="__main__":
    model_name = 'model.pth'
    model, transform_test = model_init(model_name)

    path = 'test/'
    testList = os.listdir(path)
    for file in testList:
        img = Image.open(path + file)
        predictclass, duration = resnet_predict(transform_test, model, img)
        print('time_elapsed:{}'.format(duration))
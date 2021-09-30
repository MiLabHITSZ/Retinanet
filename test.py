import torch
import torchvision.transforms as transforms
import cv2,os
from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

dict ={
        0:'ship',
        1:'grass',
        2:'net'}
print('Loading model..')
net = RetinaNet(3)
net.load_state_dict(torch.load('./checkpoint/param_50.pth'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Predicting..')
f = open('./data/1.txt','r')
save_path = './result'
if not  os.path.exists(save_path):
    os.makedirs(save_path)
lines = f.readlines()
f = open('./result.txt','w')
for line in lines:
    pic_name = line.strip().split()[0]
    print(pic_name)
    img_path = '/root/remoteSensing/zhang/R2CNN.pytorch-master/tools/datasets/ICDAR2015/test/images/{}'.format(pic_name)
    img = cv2.imread(img_path)
    w = h = 800
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x)
    loc_preds, cls_preds = net(x)

    print('Decoding..')
    encoder = DataEncoder()
    #f = open('./result.txt','a')
    try:
        boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
        # boxes,labels = encoder.decode(loc_preds.data,cls_preds.data,(w,h))
        if len(boxes) > 0:
            f.write('{} '.format(pic_name))
        for box, label, score in zip(boxes, labels, scores):
            print(score.item())
            xmin, ymin, xmax, ymax = box
            xmin_ = max(0, int(xmin.item()))
            ymin_ = max(0, int(ymin.item()))
            xmax_ = min(800, int(xmax.item()))
            ymax_ = min(800, int(ymax.item()))
            cv2.rectangle(img, (xmin_, ymin_), (xmax_, ymax_), (0, 0, 255), 1)
            f.write('{} {} {} {} {} '.format(xmin_, ymin_, xmax_, ymax_, dict[label.item()]))
            cv2.putText(img, '{}:{:.2f}'.format(dict[label.item()], score.item()), (xmin_, ymin_),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.imwrite('{}/{}'.format(save_path, pic_name), img)
    except:
        print('NULL target')

    f.write('\n')
f.close()
# 
# img = Image.open('./data/images/30_0058_0512.png')
# w = h =800
# img = img.resize((w,h))
# 
# print('Predicting..')
# x = transform(img)
# x = x.unsqueeze(0)
# x = Variable(x)
# loc_preds, cls_preds = net(x)
# 
# print('Decoding..')
# print(loc_preds.shape)
# print(cls_preds.shape)
# encoder = DataEncoder()
# boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(0), (w,h))
# #boxes,labels = encoder.decode(loc_preds.data,cls_preds.data,(w,h))
# draw = ImageDraw.Draw(img)
# for box,label in zip(boxes,labels):
#     print(dict[label.item()])
#     draw.rectangle(list(box), outline='red')
# img.save('./result1.png')

import torch
import torchvision.transforms as transforms

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
net.load_state_dict(torch.load('./checkpoint/param_19.pth'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('./data/images/54_0000_0000.png')
w = h =800
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x)
loc_preds, cls_preds = net(x)

print('Decoding..')
print(loc_preds.shape)
print(cls_preds.shape)
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(0), (w,h))
#boxes,labels = encoder.decode(loc_preds.data,cls_preds.data,(w,h))
draw = ImageDraw.Draw(img)
for box,label in zip(boxes,labels):
    print(dict[label.item()])
    print([box[0].item(),box[1].item(),box[2].item(),box[3].item()])
    draw.rectangle(list(box), outline='red')
img.save('./result1.png')

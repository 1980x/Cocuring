import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
#from model.resnet_wavedrop import *
from model.resnet import *
import pickle
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def call_bn(bn, x):
    return bn(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
def resModel(args): #resnet18
    
    if args.normalized:
       model = torch.nn.DataParallel(resnet18(end2end= False,  pretrained= False, num_classes=args.num_classes, normalized= True)).to(device)
    else:
       model = torch.nn.DataParallel(resnet18(end2end= False,  pretrained= False, num_classes=args.num_classes, normalized= False)).to(device)
    
    if   args.pretrained:
       
       checkpoint = torch.load('pretrained/res18_naive.pth_MSceleb.tar')
       pretrained_state_dict = checkpoint['state_dict']
       model_state_dict = model.state_dict()
       #pdb.set_trace()
       '''
       for name, param in pretrained_state_dict.items():
           print(name)
           
       for name, param in model_state_dict.items():
           print(name)
       '''
       for key in pretrained_state_dict:
           if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :
               print(key) 
               pass
           else:
               #print(key)
               model_state_dict[key] = pretrained_state_dict[key]

       model.load_state_dict(model_state_dict, strict = False)
       print('Model loaded from Msceleb pretrained')
    else:
       print('No pretrained resent18 model built.')
    return model   

if __name__ == '__main__':
    net = resModel(args)
    x = torch.rand(1, 3, 224,224)
    y = net(x, istrain= True)
    print(y.shape)





# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram
Implementation of Noisy robust FER training on RAFDB using Co-curing
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-05-2021
Email: darshangera@sssihl.edu.in
'''
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse, sys
import datetime

from PIL import Image

from noiserobustfer import noisyfer
import pandas as pd
import image_utils
import cv2
import argparse,random

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--raf_path', type=str, default='../data/RAFDB', help='Raf-DB dataset path.')
    
parser.add_argument('--pretrained', type=str, default='pretrained/res18_naive.pth_MSceleb.tar', help='Pretrained weights')

parser.add_argument('--resume', default='', type=str, metavar='PATH',   help='path to latest')
                        
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)

parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)

parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
                 
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='rafdb')

parser.add_argument('--noise_file', type=str, help='EmoLabel/', default='EmoLabel/train_label.txt')#0.4noise_train.txt

parser.add_argument('--T1', type=float, default=0.7, help='Threshold for confident samples')

parser.add_argument('--T2', type=float, default=0.3, help='Threshold for noisy samples')

parser.add_argument('--co_lambda_max', type=float, default=.9,          help='..based on ')
                    
parser.add_argument('--beta', type=float, default=.65, help='..based on ')                    

parser.add_argument('--n_epoch', type=int, default=40)

parser.add_argument('--num_classes', type=int, default=7)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--print_freq', type=int, default=30)

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

parser.add_argument('--num_iter_per_epoch', type=int, default=400)

parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--co_lambda', type=float, default=0.1)

parser.add_argument('--adjust_lr', type=int, default=1)

parser.add_argument('--num_models', type=int, default=2)

parser.add_argument('--model_type', type=str, help='[mlp,cnn,res]', default='res')

parser.add_argument('--save_model', type=str, help='save model?', default="False")

parser.add_argument('--save_result', type=str, help='save result?', default="True")

parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')

parser.add_argument('--warm_epochs', type=int, default=0, help='warm_epochs')

parser.add_argument('--relabel_epochs', type=int, default=40, help='relabel after this epoch')

parser.add_argument('--margin', type=float, default=0.7, help='margin for relbaelling')

parser.add_argument('--normalized', type=int, default=0, help='use normalized weights and features')

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

                        
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, basic_aug = False, ):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train_clean = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/train_label.txt'), sep=' ', header=None)
        df_train_noisy = pd.read_csv(os.path.join(self.raf_path, args.noise_file), sep=' ', header=None)
        
        df_test = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/test_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset_train_noisy = df_train_noisy[df_train_noisy[NAME_COLUMN].str.startswith('train')]
            dataset_train_clean = df_train_clean[df_train_clean[NAME_COLUMN].str.startswith('train')]
            self.clean_label = dataset_train_clean.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.noisy_label = dataset_train_noisy.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.label = self.noisy_label
            file_names = dataset_train_noisy.iloc[:, NAME_COLUMN].values
            self.noise_or_not = (self.noisy_label == self.clean_label) #By DG
        else:             
            dataset = df_test[df_test[NAME_COLUMN].str.startswith('test')]
            self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral            
            file_names = dataset.iloc[:, NAME_COLUMN].values
        
        self.new_label = [] 
        
        for label in self.label:
            self.new_label.append(self.change_emotion_label_same_as_affectnet(label))
            
        self.label = self.new_label
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
        
        
    def change_emotion_label_same_as_affectnet(self, emo_to_return):
        
        if emo_to_return == 0:
            emo_to_return = 3
        elif emo_to_return == 1:
            emo_to_return = 4
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 1
        elif emo_to_return == 4:
            emo_to_return = 2
        elif emo_to_return == 5:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 0

        return emo_to_return   
         
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image =  self.transform(image)
            #image2 =  self.transform2(image)
        return image, label, idx                         
                            
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        #print(self.indices)    
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        #print(self.num_samples)              
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            #print(label)
            # spdb.set_trace()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        #print(dataset_type)
        #pdb.set_trace()
        if dataset_type is RafDataSet:
            return dataset.label[idx]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples  
       

def main():
    
    print('\n\t\tAum Sri Sai Ram')
    print('\n\tFER with noisy annotations using Cocuring\n')
    print(args)
        
    args.noise_file = 'EmoLabel/'+noise_file
    
    if  args.dataset == 'rafdb':   
        input_channel = 3
        num_classes = args.num_classes
        init_epoch = 5
        args.epoch_decay_start = 100
    
        filter_outlier = False
        args.model_type = "res"
        
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),        
            transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                            transforms.RandomAffine(degrees=0, translate=(.1, .1),
                                                   scale=(1.0, 1.25),
                                                   resample=Image.BILINEAR)],p=0.5),
            
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            
            ])
    
        data_transforms_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
                                     
        train_dataset = RafDataSet(args.raf_path, phase = 'train', transform = data_transforms, basic_aug = True)    
        
        print('Train set size:', train_dataset.__len__())                                                                            
        test_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)    
        print('Validation set size:', test_dataset.__len__())
        
                                
    
    if args.forget_rate is None:
        forget_rate = args.noise_rate
    else:
        forget_rate = args.forget_rate
    '''
    train_sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)    
    '''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               drop_last=True,
                                               shuffle = True,  
                                               pin_memory = True) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)                                    
    # Define models
    print('building model...'+ str(args.num_models)+' with noise file: '+args.noise_file)

    #model= noiserobustfer(args, train_dataset, device, input_channel, num_classes)
    model = noisyfer(args, train_dataset, device, input_channel, num_classes)
    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    
    # evaluate models with random weights
    test_acc1, test_acc2, acc = model.evaluate(test_loader)

    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f Avg Accuracy %.4f' % (
            0, args.n_epoch, len(test_dataset), test_acc1, test_acc2, acc))

    
    
    best_acc = 0.0
    acc_list = []
    # training
    for epoch in range(0, args.n_epoch):
        if args.num_models == 1:
           acc = model.train(train_loader, epoch)
           acc = model.evaluate(test_loader)
           print(  'Epoch [%d/%d] Test Accuracy on the %s test images: %% Avg Accuracy %.4f' % ( epoch + 1, args.n_epoch, len(test_dataset), acc))
           
        elif args.num_models == 2:
             train_acc1, train_acc2, acc = model.train(train_loader, epoch)
             test_acc1, test_acc2, acc = model.evaluate(test_loader)
             print(  'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f  %% Avg Accuracy %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2,acc))
             if best_acc < acc:
                best_acc = acc
                print('best_acc:  ',best_acc)
       
        if acc > 89: 
               model.save_model(epoch, acc, args.noise_file.split('/')[-1],args.dataset)  
    print('Final best_acc:  ',best_acc, ' for ', noise_file) 
        
if __name__ == '__main__':
    main()
    
    
    
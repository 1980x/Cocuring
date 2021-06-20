# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.cnn import resModel
import numpy as np
from utils import accuracy
from loss import jointloss
import os

class noisyfer:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        self.T1 = args.T1
        self.T2 = args.T2
        self.margin = args.margin
        self.warm_epochs  = args.warm_epochs
        self.relabel_epochs = args.relabel_epochs
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2
        
        self.device = device
        
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        
        if args.model_type=="res":
            self.model1 = resModel(args)     
            self.model2 = resModel(args)

        self.model1.to(device)
        

        self.model2.to(device)
        
        if  args.resume:
            if os.path.isfile(args.resume): #for 3 models
               pretrained = torch.load(args.resume)
               pretrained_state_dict1 = pretrained['model_1']   
               pretrained_state_dict2 = pretrained['model_2']   
               model1_state_dict =  self.model1.state_dict()
               model2_state_dict =  self.model2.state_dict()
               
               loaded_keys = 0
               total_keys = 0
               for key in pretrained_state_dict1: 
                   print(key)   
                   if  ((key=='module.classifier.weight')|(key=='module.classifier.bias')):                  
                       print(key)
                       pass
                   else:    
                       model1_state_dict[key] = pretrained_state_dict1[key]
                       model2_state_dict[key] = pretrained_state_dict2[key]                       
                       total_keys+=1
                       if key in model1_state_dict and key in model2_state_dict:# and  key in model3_state_dict:
                          loaded_keys+=1
               print("Loaded params num:", loaded_keys)
               print("Total params num:", total_keys)
               self.model1.load_state_dict(model1_state_dict) 
               self.model2.load_state_dict(model2_state_dict)
               print('All 2 models loaded from ',args.resume)
        
        filter_list = ['module.classifier.weight', 'module.classifier.bias']
        base_parameters_model1 = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model1.named_parameters()))))
        base_parameters_model2 = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model2.named_parameters()))))
     
        
        self.optimizer1 = torch.optim.Adam([{'params': base_parameters_model1}, {'params': list(self.model1.module.classifier.parameters()), 'lr': learning_rate}], lr=1e-3)
        self.optimizer2 = torch.optim.Adam([{'params': base_parameters_model2}, {'params': list(self.model2.module.classifier.parameters()), 'lr': learning_rate}], lr=1e-3)
        
        self.m1_statedict =  self.model1.state_dict()
        self.m2_statedict =  self.model2.state_dict()
        self.o1_statedict = self.optimizer1.state_dict()     
        self.o2_statedict = self.optimizer2.state_dict()                                        

        self.loss_fn = loss_sai_single


        self.adjust_lr = args.adjust_lr
    
    def save_model(self, epoch, acc, noise, dataset):
        torch.save({'epoch':  epoch,
                    'model_1': self.m1_statedict,
                    'model_2': self.m2_statedict,                    
                    'optimizer1':self.o1_statedict,        
                    'optimizer2':self.o2_statedict},                  
                     os.path.join('checkpoints/'+dataset+ "/"+str(epoch)+'_noise_'+noise+"_acc_"+str(acc)[:5]+".pth")) 
        print('Models saved '+os.path.join('checkpoints/'+dataset, "/"+str(epoch)+'_noise_'+noise+"_acc_"+str(acc)[:5]+".pth")) 
        
    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        correct  = 0
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = (images).to(self.device)
                logits1 = self.model1(images) 
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()

                logits2 = self.model2(images)
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2.data, 1)
                total2 += labels.size(0)
                correct2 += (pred2.cpu() == labels).sum()
                
                avg_output = 0.5*(outputs1.data + outputs2.data) 
                _, avg_pred = torch.max(avg_output, 1)
                correct += (avg_pred.cpu() == labels).sum()
                
            acc1 = 100 * float(correct1) / float(total1)
            acc2 = 100 * float(correct2) / float(total2)
            acc = 100 * float(correct) / float(total2)
        return acc1, acc2, acc
        
    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if epoch > 0:
           self.adjust_learning_rate(self.optimizer1, epoch)
           self.adjust_learning_rate(self.optimizer2, epoch)
        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            
            if i > self.num_iter_per_epoch:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2
            
            avg_prec = accuracy(0.5*(logits1+logits2), labels, topk=(1,)) 
             
            loss1, loss2 = loss_sai_single(logits1,logits2, labels ),loss_sai_single(logits2,logits1, labels)
            
            self.optimizer1.zero_grad()
            loss1.backward(retain_graph=True)            
            self.optimizer1.step()
            
            self.optimizer2.zero_grad()
            loss2.backward()            
            self.optimizer2.step()
            
            
            # Relabel samples
            if epoch >= self.relabel_epochs:
                
                sm1 = torch.softmax(logits1 , dim = 1)
                Pmax1, predicted_labels1 = torch.max(sm1, 1) # predictions
                
                Pgt1 = torch.gather(sm1, 1, labels.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                
                sm2 = torch.softmax(logits2 , dim = 1)
                Pmax2, predicted_labels2 = torch.max(sm2, 1) # predictions
                Pgt2 = torch.gather(sm2, 1, labels.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                
                true_or_false = (Pmax1 - Pgt1 > self.margin) & (Pmax2 - Pgt2 > self.margin)#  & lowwt_indices 
                update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                label_idx = indexes[update_idx] # get samples' index in train_loader
                relabels = predicted_labels1[update_idx] # predictions where (Pmax - Pgt > margin_2)
                
                if update_idx.numel()> 0:
                   all_labels = np.array(train_loader.dataset.label)
                   all_labels[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
                   train_loader.dataset.label = all_labels
                   
            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Avg Accuracy: %.4f, Loss1: %.4f, %%, Loss2: %.4f, %%'% (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,avg_prec,  loss1.data.item() ,loss2.data.item() ))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, avg_prec
        
        
    def adjust_learning_rate(self, optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        print('******************************')
    

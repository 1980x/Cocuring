import torch 
import torch.nn.functional as F
import numpy as np
import math

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def jointloss(y_1, y_2, t, co_lambda_max = 0.9, beta = 0.65, epoch_num = 1, max_epochs = 100): #y1,y2 are p1, p2 respectively
    
    e = epoch_num
      
    e_r = 0.9 * max_epochs
     
    co_lambda = co_lambda_max * math.exp(-1.0 * beta * (1.0 - e / e_r ) ** 2) # co_lambda_max * exp(-beta (1-e/e_r)^2) based on  Noisy Concurrent Training for Efficient Learning under Label Noise(WACV 2021)
     
    loss_ce_1 = F.cross_entropy(y_1, t) 
    loss_ce =   (1 - co_lambda) * 0.5 * (loss_ce_1 )
    
    loss_kl =   co_lambda * 0.5 * (   kl_loss_compute(y_2, y_1))
        
    loss  =  (loss_kl + loss_ce).cpu() 
       
    #print('inside loss: ', e, e_r, co_lambda, loss_ce_1.detach().cpu().numpy(), loss_ce_2.detach().cpu().numpy(), loss_kl.detach().cpu().numpy(), loss.detach().cpu().numpy())
    return loss








    

    


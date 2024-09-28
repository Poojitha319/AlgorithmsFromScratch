#Importing libraries
import numpy as np
import matplotlib as plt
from .metrics import *
def Line_search_GD(x,y,epochs,batch_size,loss,lr_list):
    w = np.random.randn()
    b = np.random.randn()
    l_list = []
    w_list = []
    b_list = []
    points = 0
    ep = [i for i in range(epochs+1)]
    dw,db = 0,0
    for i in range(epochs+1):
        dw,db = 0,0
        for j in range(x.shape[0]):
            if (loss == 'mse'):
                dw += grad_w_mse(x[j],y[j],w,b)
                db += grad_b_mse(x[j],y[j],w,b)
            elif (loss == 'cross_entropy'):
                dw += grad_w_cross(x[j],y[j],w,b)
                db += grad_b_cross(x[j],y[j],w,b)
            points += 1
            if(points % batch_size == 0):
                best_w,best_b = w,b
                min_loss = 10000
                for i in range(len(lr_list)):
                    tmp_w = w - lr_list[i]*dw
                    tmp_b = b - lr_list[i]*db
                    if (loss == 'mse'):
                        loss = mse(x,y,tmp_w,tmp_b)[0]
                    elif (loss == 'cross_entropy'):
                        loss = cross_entropy(x,y,tmp_w,tmp_b)[0]
                    if (loss<min_loss):
                        min_loss = loss
                        best_w,best_b = tmp_w,tmp_b
                w,b = best_w,best_b
                dw,db = 0,0
        if (loss == 'mse'):
            print('Loss after {}th epoch = {}\n'.format(i,mse(x,y,w,b)[0]))
            l_list.append(mse(x,y,w,b)[0])
        elif (loss == 'cross_entropy'):
            print('Loss after {}th epoch = {}\n'.format(i,cross_entropy(x,y,w,b)[0]))
            l_list.append(cross_entropy(x,y,w,b)[0])
        w_list.append(w[0])
        b_list.append(b[0])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch Curve\nAlgotithm :Line Search Mini Batch Gradient Decent\nBatch Size = {}\nLoss Function = {}'.format(batch_size,loss))
    #plt.plot(ep,l_list)
    #plt.show()
    return w_list,b_list
import numpy as np
import matplotlib as plt
from .metrics import *
#Mini Batch Momentum Based Gradient Discent
def Momentum_GD(x,y,epochs,batch_size,loss,eta,lr):
    w = np.random.randn()
    b = np.random.randn()
    prev_w = 0
    prev_b = 0
    l_list = []
    w_list = []
    b_list = []
    ep = [i for i in range(epochs+1)]
    dw,db = 0,0
    points = 0
    for i in range(epochs+1):
        dw,db = 0,0
        for j in range(x.shape[0]):
            if (loss == 'mse'):
                dw += grad_w_mse(x[j],y[j],w,b)
                db += grad_b_mse(x[j],y[j],w,b)
            elif (loss == 'cross_entropy'):
                dw += grad_w_cross(x[j],y[j],w,b)
                db += grad_b_cross(x[j],y[j],w,b)
            points +=1
            if (points % batch_size == 0):
                updated_w = lr*prev_w + eta*dw
                updated_b = lr*prev_b + eta*db
                w = w - updated_w
                b = b - updated_w
                prev_w = updated_w
                prev_b = updated_b
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
    plt.title('Loss vs Epoch Curve\nAlgotithm :Mini Batch Momentum Based Gradient Decent\nBatch SIze = {}\nLearning Rate(Gamma) = {}\nEta = {}\nLoss Function = {}'.format(batch_size,lr,eta,loss))
    plt.plot(ep,l_list)
    plt.show()
    return w_list,b_list
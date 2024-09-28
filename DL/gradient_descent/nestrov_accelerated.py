#Mini Batch Nesterov accelerated Gradient Discent
import matplotlib as plt
import numpy as np
from .metrics import *
def Nesterov_GD(x,y,epochs,batch_size,loss,eta,lr):
    w = np.random.randn()
    b = np.random.randn()
    prev_w_look_ahead,prev_b_look_ahead = 0,0
    points = 0
    l_list = []
    w_list = []
    b_list = []
    ep = [i for i in range(epochs+1)]
    dw_look_ahead,db_look_ahead = 0,0
    #First Look Ahead Point
    w_look_ahead = w - lr*prev_w_look_ahead                      #W_look_ahead = w_t - lr*w_update_t-1
    b_look_ahead = b - lr*prev_b_look_ahead                      #B_look_ahead = b_t - lr*b_update_t-1
    for i in range(epochs+1):
        dw_look_ahead,db_look_ahead = 0,0
        for j in range(x.shape[0]):
            #Gradients w.r.t Look Ahead Points
            if (loss == 'mse'):
                dw_look_ahead += grad_w_mse(x[j],y[j],w_look_ahead,b_look_ahead)      
                db_look_ahead += grad_b_mse(x[j],y[j],w_look_ahead,b_look_ahead)
            elif (loss == 'cross_entropy'):
                dw_look_ahead += grad_w_cross(x[j],y[j],w_look_ahead,b_look_ahead)
                db_look_ahead += grad_b_cross(x[j],y[j],w_look_ahead,b_look_ahead)
            points += 1
            if(points % batch_size == 0):
                updated_w = lr*prev_w_look_ahead + eta*dw_look_ahead         #w_update_t = lr*w_update_t-1 + eta*gradient(w_look_ahead)
                updated_b = lr*prev_b_look_ahead + eta*db_look_ahead         #b_update_t = lr*b_update_t-1 + eta*gradient(b_look_ahead)
                w = w - updated_w                                            #W_(t+1) = w_t - w_update_t
                b = b - updated_w                                            #B_(t+1) = b_t - b_update_t
                prev_w_look_ahead = updated_w
                prev_b_look_ahead = updated_b
                #New Look Ahead point after mini batch parameter update
                w_look_ahead = w - lr*prev_w_look_ahead                     
                b_look_ahead = b - lr*prev_b_look_ahead
                dw_look_ahead,db_look_ahead = 0,0
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
    plt.title('Loss vs Epoch Curve\nAlgotithm : Nesterov Accelerated Gradient Decent\nBatch Size = {}\nLearning Rate(Gamma) = {}\nEta = {}\nLoss Function = {}'.format(batch_size,lr,eta,loss))
    plt.plot(ep,l_list)
    plt.show()
    return w_list,b_list
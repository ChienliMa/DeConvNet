import Layers
from Layers import *

import cPickle
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from matplotlib import pyplot as plt


import pdb

def Relu_nonlinear(x):
    return (T.abs_(x)+x)/2

def evaluate_lenet5( learning_rate = 0.3, batch_size =  1, n_epochs =75 ):
    print "Loading data..."
    rng = np.random.RandomState(23455)



    print "Loading params..."
    file = open('params_v5.1.pkl')
    p = cPickle.load(file)
    file.close()


    print "Sharing data..."

    print "Building architecture..."
    print "Haha =_=-b!"
    x_input = np.zeros( [1,3,32,32], dtype='float32')

    x = theano.shared( value = x_input, borrow = True )
    index = T.iscalar('index')
    print "Haha =_=-b!"
    layer0 = ConvPoolLayer( rng =rng, input = x  , image_shape = ( batch_size,3,32,32 ), 
                                            filter_shape = ( 32, 3, 5, 5), activation = Relu_nonlinear, poolsize = ( 2,2 ), 
                                            W = p[-2], b = p[-1] ) 
    print "Haha =_=-b!"
    layer1 = ConvPoolLayer( rng =rng, input = layer0.output , image_shape = ( batch_size,32,14,14 ), 
                                            filter_shape = ( 50, 32, 5, 5),   activation = Relu_nonlinear, poolsize = ( 2,2 ),
                                            W = p[-4], b = p[-3] )
    print "Haha =_=-b!"
    layer2 = ConvPoolLayer( rng =rng, input = layer1.output , image_shape = ( batch_size,50,5,5 ), 
                                             filter_shape = ( 64, 50, 5, 5), activation = Relu_nonlinear, poolsize = ( 1,1 ),
                                             W = p[-6], b = p[-5] ) 

    # 3*3*80 = 720
    layer3_input = layer2.output.flatten(2)
    print "Haha =_=-b!"
    layer3 = HiddenLayer( rng =rng, input = layer3_input, n_in = 64, n_out = 56 , activation = T.tanh,
                                            W = p[4], b = p[5] )
    print "Haha =_=-b!"
    layer4 = HiddenLayer( rng =rng, input = layer3.output, n_in = 56, n_out = 10 , activation = T.tanh,
                                            W = p[2], b=p[3] )
    print "Haha =_=-b!"
    layer5 = LogisticRegression( input = layer4.output , n_in = 10, n_out = 10, W = p[0], b=p[1] )
    print "Haha =_=-b!"
    prediction = layer5.p_y_given_x
    print "Haha =_=-b!, T.grad!!"
    grad = T.grad( prediction[0,index], x )
    
    updates = [ ( x, x + learning_rate*grad ) ]

    generate = theano.function( [index], prediction[0,index], updates = updates )

    this_out = 0

    big_map = np.zeros([32*10,32,3])
    for  i in xrange(10):    
        x.set_value(np.zeros([1,3,32,32],dtype='float32'))
        print i
        while( generate(i) < 0.72  ):
            print generate(i)

        map = x.get_value()
        map = map.reshape([3,32,32])
        map = np.transpose(map, [1,2,0 ])
        map -=map.min()
        map/=map.max()
        map*=255
        big_map[i*32:(i+1)*32,...] = map

    plt.imshow(np.asarray(big_map, dtype='uint8'))
    plt.show()


    
if __name__ == '__main__':
    evaluate_lenet5()
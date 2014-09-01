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

from load_cifa_10 import load_cifa_10

import pdb

def Relu_nonlinear(x):
    return (T.abs_(x)+x)/2

def evaluate_lenet5( learning_rate = 0.001, batch_size =  1, n_epochs =75 ):
    print "Loading data..."
    rng = np.random.RandomState(23455)



    print "Loading params..."
    file = open('params_v5.1.pkl')
    p = cPickle.load(file)
    file.close()


    print "Sharing data..."

    print "Building architecture..."
    print "Haha =_=-b!"
    x = T.tensor4('x')
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
    prediction = layer5.p_y_given_x_in
    print "Haha =_=-b!, T.grad!!"
    grad = T.grad( prediction[0,index], x )
    
    print "Compiling function..."
    ScI = theano.function( [x, index], grad )
    predict = theano.function( [x,index],layer5.p_y_given_x[0,index] )
    test_file = open( 'test_batch', 'rb')
    map = cPickle.load( test_file )
    test_file.close()
    
    test_set_x = np.asarray( map['data'], dtype = 'float32' )
    test_set_y = np.asarray( map['labels'], dtype = 'float32' )

    plt.show()

    for i in xrange(10):
        map_out = np.zeros([32*2,32*10])
        print test_set_y[i]
        for l in xrange(10):
        
            x_in = test_set_x[ i, : ].reshape( [ 1, 3, 32, 32 ] )
            x_in_1 =np.transpose( x_in[0,...], [1,2,0] )

            x_out = ScI( x_in, l )
            x_out = np.transpose(x_out.reshape([3,32,32]),[1,2,0])
            x_out = np.abs(x_out)
            x_out = x_out.max(axis=2)
            map_out[32:,l*32:(l+1)*32]=x_out


            #map_out = np.asarray(map_out,dtype='uint8')

        print "haha"
        plt.imshow( map_out,cmap="Greys_r" )
        plt.show()



    
if __name__ == '__main__':
    evaluate_lenet5()
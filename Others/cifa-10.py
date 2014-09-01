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

from load_cifa_10 import load_cifa_10


import pdb

def Relu_nonlinear(x):
    return (T.abs_(x)+x)/2

def evaluate_lenet5( learning_rate = 0.001, batch_size =  50, n_epochs =75 ):
    print "Loading data..."
    rng = np.random.RandomState(23455)

    train_set_x, train_set_y, valid_set_x, valid_set_y = load_cifa_10()

    n_train_batches = train_set_x.shape[0]/batch_size
    n_valid_batches = valid_set_x.shape[0]/batch_size

    print "Pre-processing set_x..."
    # preproessing 


    print "Sharing data..."

    print "Building architecture..."
    x = T.matrix( 'x' )


    layer0 = ConvPoolLayer( rng =rng, input = x.reshape( ( batch_size,3,32,32 ) )  , image_shape = ( batch_size,3,32,32 ), 
                                            filter_shape = ( 32, 3, 5, 5), activation = Relu_nonlinear, poolsize = ( 2,2 ) ) 

    layer1 = ConvPoolLayer( rng =rng, input = layer0.output , image_shape = ( batch_size,32,14,14 ), 
                                            filter_shape = ( 50, 32, 5, 5),   activation = Relu_nonlinear, poolsize = ( 2,2 ) )

    layer2 = ConvPoolLayer( rng =rng, input = layer1.output , image_shape = ( batch_size,50,5,5 ), 
                                             filter_shape = ( 64, 50, 3, 3), activation = Relu_nonlinear, poolsize = ( 2,2 ) ) 

    # 3*3*80 = 720
    layer3_input = layer2.output.flatten(2)

    layer3 = HiddenLayer( rng =rng, input = layer3_input, n_in = 64, n_out = 64 , activation = T.tanh)

    layer4 = HiddenLayer( rng =rng, input = layer3.output, n_in = 64, n_out = 10 , activation = T.tanh )

    layer5 = LogisticRegression( input = layer4.output , n_in = 10, n_out = 10 )

    cost = layer5.negative_log_likelihood(y)


    print "Compiling function..."
    validate_model = theano.function( [ x, y ] , layer5.errors(y) )

    # create a list of all model parameters to be fit by gradient descent
    params = layer5.params+ layer4.params+ layer3.params + layer2.params + layer1.params + layer0.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function( [ x, y ], cost, updates=updates )

    ###############
    # TRAIN MODEL #
    ###############
    print 'Start training...'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.996  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            x_in = train_set_x[ minibatch_index*batch_size:(minibatch_index+1)*batch_size,... ]
            y_in = train_set_y[ minibatch_index*batch_size:(minibatch_index+1)*batch_size,... ]

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model( minibatch_index )

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]

                                     
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print "Saving weight"
    weights = [ param.get_value() for param in params ]
    f = file( 'params.pkl', 'wb' )
    cPickle.dump( weights, f )
    f.close()
    print "Done!"

if __name__ == '__main__':
    evaluate_lenet5()
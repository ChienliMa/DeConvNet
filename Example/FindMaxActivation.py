import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import scipy
from scipy import io

import heapq
from heapq import *
import pdb
import gzip
import cPickle


class MaxActivationFinder( object ):
    """
    Illurstration
    """

    def __init__( self, num_of_max_activation , patches_per_sample, patch_size,kernel_index ):
        """
        Illurstration
        patch_size: tuple
        num_of_max_activation : int
        """

        self.num_of_max_activation =   num_of_max_activation
        self.patch_w, self.patch_h = patch_size
        self.patches_per_sample = patches_per_sample
        # pre-insert something int he heap
        self.heap = [ (-100, ) for i in xrange( num_of_max_activation ) ]


        # load your sample here
        f = gzip.open( 'mnist.pkl.gz' , 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        valid_set_x, valid_set_y = valid_set
        self.kernel_index = kernel_index
        self.data = valid_set_x

        def func():
            """
            def your function here to find the activation,
            use patches as input instead of the whole sample,
            do some careful calculatio about the input size
            remember to load the kernel before the definition of this function

            Params : input : input used to calculate specific activatrion with shape of ( 1, channel, patch_w, patch_h)
            Tpye : input : numpy.array

            """ 

            # load your kernels below
            K_index = 0
            # something wrong with the input,  we have to train this CNN again
            print "loading sample"
            map = scipy.io.loadmat('CNN')
            
            layer0_W = theano.shared(  map[ 'layer0_W' ])
            layer0_b = theano.shared(  map[ 'layer0_b' ].reshape( [20,] ))

            first_kernel = map[ 'layer1_W' ][ kernel_index,...].reshape( [1, 20, 5, 5] )
            first_bias = map[ 'layer1_b' ][0]
            layer1_W = theano.shared(  first_kernel)
            layer1_b = theano.shared(  first_bias)

            map = 0

            # now below is the implementation of the second layer  activation of lenet5
            # we find the  max activation of the first kernel of the 2th layer
            x = T.tensor4( 'x' )

            print "biuilding function"
            # Layer 0 
            poolsize = (2,2)
            Layer0_conv_out = conv.conv2d(input=x, filters=layer0_W, 
                                                                    image_shape = (1,1,28,28), filter_shape = (20,1,5,5))

            Layer0_pooled_out = downsample.max_pool_2d(input=Layer0_conv_out,
                                            ds=poolsize, ignore_border=True)
            Layer0_output = T.tanh(Layer0_pooled_out + layer0_b.dimshuffle('x', 0, 'x', 'x'))

            # Layer 1 "Still need some modification"
            Layer1_conv_out = conv.conv2d( input = Layer0_output, filters = layer1_W,
                                                                    image_shape = (1,20,12,12), filter_shape=(1,20,5,5) )

            Layer1_activation = T.tanh(Layer1_conv_out + layer1_b.dimshuffle('x', 0, 'x', 'x'))

            get_activation = theano.function( [ x ], Layer1_activation)

            return get_activation
        self.func = func()

    def FindMaxActivation( self ):
        """
        in this function, using heap, we maintian the patches that 
        yeild max activation value and store them in ( activation, patches)
        Illustration
        """
        data = self.data
        num_of_samples = 10000
        channel = 1
        sample_w = 28
        sample_h = 28

        patch_w = 14
        patch_h = 14

        print ( "Totally %d patches" ) % num_of_samples
        patch_count = 0
        for sample_index in xrange( num_of_samples ):
            #load one sample 
            this_sample = data[ sample_index,... ].reshape([1,1,28,28])
            
            heappushpop( self.heap , ( np.sum(self.func( this_sample )), this_sample ) )
            patch_count += 1
            print "sample_count", patch_count
        # finished fin max activation, save it
        
    def save( self ):
        f = open(  'MaxHeap.pkl', 'w' )
        cPickle.dump( self.heap, f )
        f.close()

def FindMaxActivation():
    """
    hehe
    """
    heap = []
    for kernel_index in xrange(20):
        print ( "the %d st kernel " ) % kernel_index
        Finder = MaxActivationFinder( num_of_max_activation = 10, patches_per_sample = 6, patch_size = (14, 14), kernel_index = kernel_index )
        Finder.FindMaxActivation()
        heap.append( Finder.heap )

    f = open(  'MaxHeap20.pkl', 'w' )
    cPickle.dump( heap, f )
    f.close()
            

if __name__ == "__main__":
    FindMaxActivation()




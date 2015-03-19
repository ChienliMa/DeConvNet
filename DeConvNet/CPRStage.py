"""
This is a file that contain two kind of stage: "Up" for forward CNN, "Down"for DeConv
each stage contain a conv layer, a pool layer, a rectification layer
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from max_pool import max_pool_2d, max_uppool_2d

# linear activation function
def linear( input ):
    return input

class CPRStage_Up( object ):
    """
    A class to discribe one stage in the forward CNN.
    Each stage include a conv layer, a pooling layer and a rectification layer
    """
    def __init__(self, image_shape, filter_shape, poolsize, W = None, b = None, activation = linear):
        """
        params :image_shape: Input size ( batch_size, channel, weight, height )
        type :image_shape: tuple with length of 4

        params :filter_shape: filter size ( channel_out, channel_in, weight, height )
        type :filter_shape: tuple with length of 4

        params :poolsize : poolsize 
        type :poolsize:  int 

        params :W: filter in 4 dimension array ( channel_out, channel_in, weight, height )
        type :W: numpy.ndarray

        params :b: bias vector ( channel_out )
        type :b: numpy.ndarray
        """
        assert W != None
        assert b != None
        assert image_shape[1] == filter_shape[1]

        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.activation = activation

        self.W = theano.shared( np.asarray(W,dtype=theano.config.floatX)  )
        self.b = theano.shared( np.asarray(b,dtype=theano.config.floatX)  )

        x = T.tensor4('x')
        conv_out = conv.conv2d( input = x  , filters = W, filter_shape = filter_shape,
                                                image_shape = image_shape, border_mode = 'valid' )
        output = conv_out + self.b.dimshuffle('x',0,'x','x')
        self.conv = theano.function( [ x ] , output, allow_input_downcast=True )
   
    def GetOutput( self, input = input ):
        """
        params : input : input image usually with one batch ( batch_size, channel, weight,height)
        type : input : numpy.ndarray

        """
        if input.dtype == theano.config.floatX:
            conv_out = self.conv( input )
        else:
            conv_out = self.conv( input.astype(theano.config.floatX) )

        if( conv_out.shape[1] == 1 ):
          conv_out = conv_out.reshape( conv_out.shape[2:4] )
        else: 
          conv_out = conv_out.reshape( conv_out.shape[1:4] )

        pooled_out, switch_map = max_pool_2d( conv_out , poolsize = self.poolsize)

        if( pooled_out.ndim == 2 ):
          pooled_out = pooled_out.reshape( [1,1] + list(pooled_out.shape))
        else:
          pooled_out = pooled_out.reshape( [1] + list(pooled_out.shape))

        output = self.activation( pooled_out )

        return np.asarray(output, dtype = theano.config.floatX), switch_map


class CPRStage_Down( object ):
    """
    A class to discribe one stage in the deconvnet.
    Each stage include an uppooling layer, a deconv layer and a rectification layer
    """

    def __init__( self, image_shape, filter_shape, poolsize, W, b, activation = linear):
        """
        params :image_shape: the size of uppooled input ( batch_size, channel, weight, height )
        type :image_shape: tuple with length of 4

        params :filter_shape: filter size ( channel_out, channel_in, weight, height )
        type :filter_shape: tuple with length of 4

        params :poolsize : poolsize 
        type :poolsize:  int

        params :W: *Original* filter in 4 dimension array ( channel_out, channel_in, weight, height )
        type :W: numpy.ndarray

        params :b: bias vector ( channel_out )
        type :b: numpy.ndarray
        """

        assert W != None
        assert b != None
        assert image_shape[1] == filter_shape[1]


        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.activation = activation

        W = W.transpose([1,0,2,3])
        W = W[:,:,::-1,::-1]
        self.W = theano.shared( np.asarray(W,dtype=theano.config.floatX) )
        self.b = theano.shared( np.asarray(b,dtype=theano.config.floatX)  )
 
        x = T.tensor4('x')
        conv_out = conv.conv2d( input = x  , filters = W, filter_shape = filter_shape,
                                                image_shape = image_shape, border_mode = 'full' )
        self.conv = theano.function( [ x ] , conv_out, allow_input_downcast=True )

    def GetOutput( self, input, switch_map ):
        """
       params : input : input image usually with one batch ( batch_size, channel, weight,height)
       type : input : numpy.ndarray

       params :switch_map: map that stores the exact location of max value ( see max_pool.max_uppool_2d())
       type :switch_map: numpy.ndarray

       params :activation:  activation function
       type :activation: function
        """

        input = self.activation( input )        
        
        if( input.shape[1] == 1 ):
          input = input.reshape( input.shape[2:4] )
        else: 
          input = input.reshape( input.shape[1:4] )
          
        up_pooled_out = max_uppool_2d( input, switch_map, poolsize = self.poolsize )

        if( up_pooled_out.ndim == 2 ):
          up_pooled_out = up_pooled_out.reshape( [1,1] + list(up_pooled_out.shape))
        else:
          up_pooled_out = up_pooled_out.reshape( [1] + list(up_pooled_out.shape))

        if up_pooled_out.dtype == theano.config.floatX:
            conv_out = self.conv( up_pooled_out )
        else:
            conv_out = self.conv( up_pooled_out.astype(theano.config.floatX) )
            
        output = conv_out#self.activation( conv_out )

        return np.asarray(output, dtype = theano.config.floatX )









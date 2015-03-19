class Stage():
    """
    A class to discribe one stage in the CNN.
    Each stage include a conv layer, a pooling layer and a rectification layer
    """
    def __init__(self, in_shape, out_shape, W, b, conv_stride=(1, 1)):  
        """
        param :in_shape: Shape of the input of the conv laver
        type  :in_shape: List with length of 4 (b, c, 0, 1)

        param :out_shape: Shape of the output of the pooling layer
        type  :out_shape: List with length of 4 (b, c, 0, 1)

        params :stirde : conv_stride 
        type :stride : tuple of 2 int, which represent the dx and dy in conv stride 

        params :W: filter in 4 dimension array ( channel_out, channel_in, weight, height )
        type :W: numpy.ndarray

        params :b: bias vector ( channel_out )
        type :b: numpy.ndarray
        """
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.W = theano.shared(np.asarray(W, dtype=theano.config.floatX))
        self.b = theano.shared(np.asarray(b, dtype=theano.config.floatX))
        self.W1 = theano.shared(np.asarray(W[:,:,::-1,::-1], dtype=theano.config.floatX))

        # constuct conv and deconv function
        x = T.tensor4('x')
        conv_out = conv.conv2d(input = x  , filters = self.W, 
                                    subsample = conv_stride, border_mode = 'valid')
        output = conv_out + self.b.dimshuffle('x',0,'x','x')

        # convolution function
        self.conv = theano.function([ x ] , output, allow_input_downcast = True)

        deconv_out = conv.conv2d(input = x - self.b.dimshuffle('x',0,'x','x'), 
                                    filters = self.W1, border_mode = 'full')

        # the deconvolution function
        self.deconv = theano.function([ x ] , deconv_out, allow_input_downcast = True)    

    def Forward(self, input):
        """
        Calculate the forward output of one stage.( IN -> CONV -> POOL -> ACTIVATE -> OUT )

        param :img: input images
        type  :img: Numpy 4d array with each dimension represent (b, c, 0, 1)
        """
        assert input.shape == self.in_shape

        if input.dtype == theano.config.floatX:
            conv_out = self.conv(input)
        else:
            conv_out = self.conv(input.astype(theano.config.floatX))

        # reshape to fit the API of uppool
        b, c, h, w = input.shape
        input.reshape( [b*c, h, w] ) 

        pooled_out, switch_map = max_pool_2d(conv_out , poolsize = self.poolsize)

        # reshape to fit the API of Forward
        up_h, up_w = pooled_out.shape[-2:]
        pooled_out = pooled_out.reshape( [b, c, up_h, up_w ] )

        act_output = self.activation(pooled_out)
        output = self.resize( act_output, self.out_shape )
        return np.asarray(output, dtype = theano.config.floatX), switch_map

    def Backward(self, input, switch_map):
        """
        Calculae te backword output of one stage( IN -> UPOOL -> DECONV -> OUT )

        param :img: input images
        type  :img: Numpy 4d array with each dimension represent (b, c, 0, 1)
        """
        assert input.shape == self.out_shape

        input = self.activation(input)        

        # reshape to fit the API of uppool
        b, c, h, w = input.shape
        input.reshape( [b*c, h, w] ) 
          
        up_pooled_out = max_uppool_2d(input, switch_map, poolsize = self.poolsize)

        # reshape to fit the API of Backwork
        up_h, up_w = up_pooled_out.shape[-2:]
        up_pooled_out = up_pooled_out.reshape( [b, c, up_h, up_w ] )

        if up_pooled_out.dtype == theano.config.floatX:
            conv_out = self.deconv(up_pooled_out)
        else:
            conv_out = self.deconv(up_pooled_out.astype(theano.config.floatX))
            
        # upsample, to mimic the stride of deconv
        b, c, x, y = deconv_out.shape
        dx, dy = self.conv_stride

        upsampled_out = np.zeros([b,c, dx * x, dy * y ) 
        upsampled_out[:,:,::dx,::dy] = deconv_out

        output = self.resize( upsampled_out, self.in_shape )
        return np.asarray(output, dtype = theano.config.floatX)

    def resize(self, img, img_size):
        """
        Resize the img by cutting large image or expanding small image
        to fit the size of img_size

        param :img: input images
        type  :img: Numpy 4d array with each dimension represent (b, c, 0, 1)

        param :img_shape: Expected image shape
        type  :img_shape: Tuple or list of 4 int, represents the epected size of image
        """
        assert img.dim == 4
        assert len(img_size) == 4

        orig_shape = img.shape
        output = np.zeros(img_size)
        h = min(orig_shape[2], img_size[2])
        w = min(orig_shape[3], img_size[3])

        output[:,:,:h,:w] = img[:,:,:h,:w]
        return output
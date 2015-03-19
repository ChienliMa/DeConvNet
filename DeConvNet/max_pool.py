import  numpy as np

def max_pool_2d( input, poolsize = 2):
    """
    A function executing max pooling with each feature map
    
    :type input: numpy.ndarray with 2 ot 3 dimension
    :params input: input feature maps ( #num_of_map, #width, #height)

    :type poolsize: int
    "param poolsize":  the downsampling (pooling) factor

    """
    assert input.ndim == 2 or input.ndim == 3

    if ( input.ndim ==3 ):
        num_of_map, width, height = input.shape

        output = np.zeros( ( num_of_map, width / poolsize, height / poolsize ) )
        switch_map = np.zeros( ( num_of_map, width / poolsize, height / poolsize ) )

        for map_index in xrange( input.shape[0] ):
            output[ map_index , :,  : ] , switch_map[ map_index, : , : ] = max_pool_2d( input[ map_index , : , : ] , poolsize )

    else:
        output = np.zeros( [ size / poolsize for size in input.shape ] )
        switch_map = np.zeros( [ size / poolsize for size in input.shape ] )

        for blocks_x in xrange( output.shape[0] ):
            for blocks_y in xrange( output.shape[1]):
                this_block = input[ ( blocks_x  ) * poolsize : ( blocks_x + 1  ) * poolsize,
                                               ( blocks_y  ) * poolsize : ( blocks_y + 1  ) * poolsize ]
                this_block = this_block.reshape( poolsize ** 2 )
                max_value = this_block.max()
                # np.where return a tuple yet what we need is its first element
                # whicw means if there are several max value , we take the first one
                max_index = np.where( this_block == max_value )[0][0]

                output[ blocks_x, blocks_y ] = max_value
                switch_map[ blocks_x, blocks_y ] = max_index

        return output , switch_map

    return output, switch_map


def max_uppool_2d( input, switch_map, poolsize = 2):
    """
    A function executing max uppooling with each feature map
    
    :type input: numpy.ndarray with 3 dimension
    :params input: input feature maps ( #num_of_map, #width, #height)

    :type switch_map: numpy.ndarray with 2 or 3 dimension
    :params switch_map: maps that store original location of max value in each block

    :type poolsize: int
    "param poolsize":  the downsampling (pooling) factor
    """
    assert input.shape[-2:] == switch_map.shape[-2:]
    assert input.ndim == 2 or input.ndim == 3

    if input.ndim == 3:
        num_of_map, w_in, h_in = input.shape
        w_out = w_in * poolsize;
        h_out = h_in * poolsize;

        output = np.zeros( ( num_of_map, w_out, h_out ) )
        for map_index in xrange( output.shape[0]):
            output[ map_index, : ,: ] = max_uppool_2d( input[ map_index, : , : ],
                                                       switch_map[ map_index, : , : ], 
                                                        poolsize )
    else:
        output = np.zeros( [size*poolsize for size in input.shape] )
        for blocks_x in xrange( input.shape[0] ):
            for blocks_y in xrange( input.shape[1] ):
                index = switch_map[ blocks_x, blocks_y ]
                x_bias = int( index / poolsize )
                y_bias = int( index % poolsize )
                output[ blocks_x * poolsize + x_bias, blocks_y * poolsize + y_bias ] = input[ blocks_x, blocks_y ]

    return output 
import os
import sys
import numpy as np
import cPickle
from matplotlib import pyplot as plt
from utils import tile_raster_images

# in order to import customeized classes
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
from CPRStage import CPRStage_Up,CPRStage_Down
import theano
theano.config.floatX = 'float32'
def activation( a ):
    return ( np.abs(a) + a ) /2

def example1():
    """
    In this example, I visulize what the 3rd layer 'see' altoghther.
    By set none of ferature maps in 3rd layer to zero.
    """

    print "Loading model..."
    model_file = open( 'params.pkl', 'r')
    params = cPickle.load( model_file )
    model_file.close()

    layer0_w = params[-2]
    layer0_b = params[-1]
    layer1_w = params[-4]
    layer1_b = params[-3]
    layer2_w = params[-6]
    layer2_b = params[-5]

    # fordward
    up_layer0 = CPRStage_Up( image_shape = (1,3,32,32), filter_shape = (32,3,5,5),
                            poolsize = 2 , W = layer0_w, b = layer0_b, 
                            activation = activation)
                            
    up_layer1 = CPRStage_Up( image_shape = (1,32,14,14), filter_shape = (50,32,5,5), 
                            poolsize = 2,W = layer1_w, b = layer1_b ,
                            activation = activation)
                            
    up_layer2 = CPRStage_Up( image_shape = (1,50,5,5), filter_shape = (64,50,5,5), 
                            poolsize = 1,W = layer2_w, b = layer2_b ,
                            activation = activation)
    # backward
    down_layer2 = CPRStage_Down( image_shape = (1,64,1,1), filter_shape = (50,64,5,5), 
                                poolsize = 1,W =layer2_w, b = layer2_b,
                                activation = activation)
                                
    down_layer1 = CPRStage_Down( image_shape = (1,50,10,10), filter_shape = (32,50,5,5), 
                                poolsize = 2,W =layer1_w, b = layer1_b,
                                activation = activation)
                                
    down_layer0 = CPRStage_Down( image_shape = (1,32,28,28), filter_shape = (3,32,5,5), 
                                poolsize = 2,W = layer0_w, b = layer0_b,
                                activation = activation)


    # load sample images
    print 'Loading sample images...'
    f = open( 'SubSet25.pkl', 'r' )
    input = cPickle.load( f )
    f.close()

    output = np.ndarray( input.shape )
    num_of_sam = input.shape[0]
    print 'Totally %d images' % num_of_sam

    for i in xrange(num_of_sam):
        print '\tdealing with %d image...' % (i+1)
        l0u , sw0 = up_layer0.GetOutput( input[i].reshape(1,3,32,32) )
        l1u  , sw1 = up_layer1.GetOutput( l0u )
        l2u  , sw2 = up_layer2.GetOutput( l1u )
        
        l2d = down_layer2.GetOutput( l2u, sw2 )
        l1d = down_layer1.GetOutput( l2d, sw1 )
        l0d = down_layer0.GetOutput( l1d , sw0 )
        output[i] = l0d
        
    # from bc01 to cb01
    input = np.transpose( input, [ 1, 0, 2, 3 ])         
    output = np.transpose( output, [ 1, 0, 2, 3 ])
    
    # flatten
    input = input.reshape( [ 3, 25, 32*32 ])
    output = output.reshape( [ 3, 25, 32*32 ])

    # transform to fit tile_raster_images    
    input = tuple( [ input[i] for i in xrange(3)] + [None] )    
    output = tuple( [ output[i] for i in xrange(3)] + [None] )   
    
    input_map = tile_raster_images( input, img_shape = (32,32), tile_shape = (5,5), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
                                    
    output_map = tile_raster_images( output, img_shape = (32,32), tile_shape = (5,5), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
    
    bigmap = np.append( input_map, output_map, axis = 1 )      

    plt.imshow(bigmap)
    plt.show()

if __name__ == "__main__":
    example1()

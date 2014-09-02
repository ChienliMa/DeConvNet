#-*- coding: utf-8 -*- 
import os
import sys
import numpy as np
import cPickle
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
from utils import tile_raster_images
from heapq import *

# in order to import customeized classes
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
from CPRStage import CPRStage_Up,CPRStage_Down
from Layers import ConvPoolLayer,relu_nonlinear


def activation( a ):
    return ( np.abs(a) + a ) /2
    
    
class Pairs( object ):
    """
    Customized class to avoid ambigous compare result of tuples
    """
    def __init__( self, activation, input ):
        self.act = activation
        self.sam = input
        
    def __lt__( self, obj ):
        """
        Customized __lt__ to avoid compare input
        """
        return self.act < obj.act

class DeConvNet( object ):
    """
    for one-time use
    """
    def __init__( self ):

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


        # compile theano function for efficient forward propagation
        x = T.tensor4('x')
 
        layer0 = ConvPoolLayer( input = x, image_shape = (1,3,32,32), 
                                filter_shape = (32,3,5,5), W = layer0_w,
                                b = layer0_b, poolsize=(2, 2), 
                                activation = relu_nonlinear)
                                
        layer1 = ConvPoolLayer( input = layer0.output, image_shape = (1,32,14,14), 
                                filter_shape = (50,32,5,5), W = layer1_w, 
                                b = layer1_b, poolsize=(2, 2), 
                                activation = relu_nonlinear)
        
        layer2 = ConvPoolLayer( input = layer1.output, image_shape = (1,50,5,5), 
                                filter_shape = (64,50,5,5), W = layer2_w, 
                                b = layer2_b, poolsize=(1, 1), 
                                activation = relu_nonlinear) 
        print "Compiling theano.function..."
        self.forward = theano.function( [x], layer2.output )
                               
        # Stages that ZUCHENG DeConvNet
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

        self.Stages = [ up_layer0, up_layer1, up_layer2,
                           down_layer2, down_layer1, down_layer0]
        
    def DeConv( self, input, kernel_index ):

        assert kernel_index != None        
        
        l0u , sw0 = self.Stages[0].GetOutput( input.reshape(1,3,32,32) )
        l1u  , sw1 = self.Stages[1].GetOutput( l0u )
        l2u  , sw2 = self.Stages[2].GetOutput( l1u )
        
        # only visulize selected kernel
        l2u[0,:kernel_index,...]*=0
        l2u[0,kernel_index+1:,...]*=0
        
        
        l2d = self.Stages[3].GetOutput( l2u, sw2 )
        l1d = self.Stages[4].GetOutput( l2d, sw1 )
        l0d = self.Stages[5].GetOutput( l1d , sw0 )
  
        return l0d
        
    
def findmaxactivation( Net, samples, num_of_maximum, kernel_list):
    """
    function，给定forward和数据集，加上model，所需的参数，
    返回list of tuple，包含最大激活值
    """    

    Heaps = { kernel_i: [ Pairs( -100, -i) for i in xrange(num_of_maximum)]\
                    for kernel_i in  kernel_list  }

    
    index = 0
    print "totally %d samples" % samples.shape[0]    
    for sam in samples:
        index += 1
        print "pushpop %d sample" % index
        # from 3-dim to 4-dim
        sam = sam.reshape((1,)+sam.shape )      
        activate_value = Net.forward(sam).flatten()
        for kernel_i in kernel_list:
            heappushpop( Heaps[kernel_i], Pairs( activate_value[kernel_i], sam ))

    return Heaps
    

    

def Find_cifa_10():
    """
    balabala
    """
    
    sam_file = open('SubSet1000.pkl', 'r')
    samples = cPickle.load( sam_file )
    sam_file.close()
    
    Net = DeConvNet()
    
    kernel_list = [ 2,23,60,12,45,9 ]    
    
    Heaps = findmaxactivation( Net, samples, 9, kernel_list )
    bigbigmap = None
    for kernel_index in Heaps:
        print 'kernelindex',kernel_index
        heap = Heaps[kernel_index]
        this_sams = []
        this_Deconv = []
        for pairs in heap:
            this_sam = pairs.sam
            this_sams.append( this_sam.reshape([3,32,32]) )
            this_Deconv.append( Net.DeConv( this_sam, kernel_index ).reshape([3,32,32]) )
        
        this_sams = np.array( this_sams )
        this_sams = np.transpose( this_sams, [ 1, 0, 2, 3 ])
        this_sams = this_sams.reshape( [ 3, 9, 32*32 ])
        this_sams = tuple( [ this_sams[i] for i in xrange(3)] + [None] )    
        
        this_Deconv = np.array( this_Deconv )
        this_Deconv = np.transpose( this_Deconv, [ 1, 0, 2, 3 ])
        this_Deconv = this_Deconv.reshape( [ 3, 9, 32*32 ])
        this_Deconv = tuple( [ this_Deconv[i] for i in xrange(3)] + [None] )

        this_map = tile_raster_images( this_sams, img_shape = (32,32), tile_shape = (3,3), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
        this_Deconv = tile_raster_images( this_Deconv, img_shape = (32,32), tile_shape = (3,3), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
        this_pairmap = np.append( this_map, this_Deconv, axis = 1)

        if bigbigmap == None:
            bigbigmap = this_pairmap
        else:
            bigbigmap = np.append(bigbigmap, this_pairmap, axis=1)
    plt.imshow(bigbigmap)
    plt.show()
        
        
        


if __name__ == "__main__":
    Find_cifa_10()
    
    
    



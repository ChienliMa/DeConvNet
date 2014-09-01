import cPickle
import numpy as np

datapath = '/media/chienli/Bank/Dataset/cifa-10/'

def load_cifa_10():
    """
    a function to load cifa_10 dataset and return
    ( train_set_x, train_set_y, test_set_x, test_set_y )
    """
    train_set_x = np.ndarray([ 50000, 3072 ])
    train_set_y = np.ndarray( [50000] )

    batch_size = 10000
    for i in xrange(5):
        batch = open( datapath + "data_batch_"+str(i+1), 'rb')
        map = cPickle.load( batch )
        batch.close()
        train_set_x[ i*batch_size : (i+1)*batch_size , : ] = np.asarray( map[ 'data' ], dtype = 'float32' )
        train_set_y[ i*batch_size : (i+1)*batch_size ] = np.asarray( map[ 'labels' ], dtype = 'float32' )

    test_file = open( datapath + 'test_batch', 'rb')
    map = cPickle.load( test_file )
    test_file.close()
    
    test_set_x = np.asarray( map['data'], dtype = 'float32' )
    test_set_y = np.asarray( map['labels'], dtype = 'float32' )
    

    return train_set_x, train_set_y, test_set_x, test_set_y



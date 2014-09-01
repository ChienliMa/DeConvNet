import numpy as np
import unittest
from max_pool import max_pool_2d, max_uppool_2d

class test_pooling( unittest.TestCase ):
    
    def setUp( self ):
        pass

    def tearDown( self ):
        pass

    def test_max_pool( self ):
        input = np.asarray( [ [ [0,0,1,1],
                                    [0,0,1,1],
                                    [2,2,3,3],
                                    [2,2,3,3]],
                                    [ [1,2,-1,-2],
                                    [3,4,-3,-4],
                                    [0,2,1,1],
                                    [3,5,0,1] ] ] )
        desired_output = np.asarray( [ [ [0,1],
                                                    [2,3]],
                                                    [ [4,-1],
                                                    [5,1] ] ] )
        desired_switch_map = np.asarray( [ [ [0,0],
                                                        [0,0]],
                                                        [ [3,0],
                                                        [3,0] ] ] )
        poolsize = 2
        actual_output, actual_switch_map = max_pool_2d( input, poolsize )
        assert ( desired_output == actual_output ).all()
        assert ( desired_switch_map == actual_switch_map ).all()


    def test_max_uppool( self ):
        input = np.asarray( [ [ [0,1],
                                        [2,3]],
                                    [ [4,-1],
                                        [5,1] ] ] )
        switch_map = np.asarray( [ [ [0,0],
                                                [0,0]],
                                                [ [3,0],
                                                [3,0] ] ] )
        desired_output = np.asarray([[[0,0,1,0],
                                                    [0,0,0,0],
                                                    [2,0,3,0],
                                                    [0,0,0,0]],
                                                    [[0,0,-1,0],
                                                    [0,4,0,0],
                                                    [0,0,1,0],
                                                    [0,5,0,0]]])
        poolsize = 2;
        assert ( desired_output == max_uppool_2d( input, switch_map, poolsize ) ).all()

if __name__ =='__main__':  
    unittest.main()  
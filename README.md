DeConvNet
===========

Personal implementation of DeConvNet, used to visualize CNN. Follow Pro.Zealer's paper:
    
>>[Visualizing and Understanding Convolutional Networks](http://arxiv.org/pdf/1311.2901v3.pdf)<br />



##What in here?
>###CPRStages_up(down)
>One stage consists of one pooling layer(2D, not overlapping), one convolution layer(2D not overlapping) and one activation function. The series of up_stage and down_stage are totally inverse. For more detail, look at the code in DeCoovNet/DeConvNet/CPRStage.



>###Examples:
Two examples inllustrate how to use this to visulize CNN. Admittedlly, currentle this is not very easy to use. My fault :-(
    
>I have trained a cifa-10 CNN with 3 conv layers and a 70% AR. And the paremeters stored in DeCoonvNet/Exampls/Params.pkl . Here I will show how this work using this structure.

>####Example1
>Visulize what the kernels in 3rd layer 'see' together by not setting other any output map to zero.( pictures were randomly picked )

>![EX1](https://raw.githubusercontent.com/ChienliMa/DeCoonvNet/master/Example1.png "EX1")  



>####Example2
> Use heaps and a simple forward conv net to find samples that yield max activation value in 2,23,60,12,45,9th kernel. And then visualize what those kernels see separatly, using a deconvnet, by setting other feature map to zeros. 
More specifically, the 23rd kernel is sensitive to boats, the 12nd and 45th kernel respones to cars strongly and the 9th kernel like airplanes very much.

>![EX2](https://raw.githubusercontent.com/ChienliMa/DeCoonvNet/master/Example2.png "EX2")  


###Future Improvemence:
>1. 完善API, less reshape is needed, 输入三维矩阵，在内使用theano接口，在外就算了
>2. convert this into a theano op and push it to theano。master

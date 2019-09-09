import torch

from torch import nn
from torch.autograd import Function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'dist/cpools-0.0.0-py3.6-linux-x86_64.egg'))
import top_pool, bottom_pool, left_pool, right_pool

class TopPoolFunction(Function):
    # first call
    # train.py :      nnet = NetworkFactory
    # NetworkFactory inherited from kp
    # kp.tl_cnvs is the output of make_tl_layer(256)
    # tl_pool(256)
    # go to pool(256,TopPool,LeftPool)
    # go to TopPool
    # go to TopPoolFunction in __init__.py
    # go to top_pool in cpools/src/setup.py forward 是将一个热力图的每一列最大值的row提取出来
    @staticmethod
    def forward(ctx, input):
        # input 是热力图
        # ctx can be seen as the context in which this Function is running. You will get an empty one during the forward that only contains helper functions. The same ctx will be passed to the backward function so you can use it to store stuff for the backward.
        # It is similar to the self argument when working with python classes.
        output = top_pool.forward(input)[0]
        # forward giving a height x width heatmap, and return a tensor with shape(width,)
        # for each col count which row has the largest value
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = top_pool.backward(input, grad_output)[0]
        return output

class BottomPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = bottom_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = bottom_pool.backward(input, grad_output)[0]
        return output

class LeftPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = left_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = left_pool.backward(input, grad_output)[0]
        return output

class RightPoolFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = right_pool.forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input  = ctx.saved_variables[0]
        output = right_pool.backward(input, grad_output)[0]
        return output

class TopPool(nn.Module):
    def forward(self, x):
        return TopPoolFunction.apply(x)
    # first call
    # train.py :      nnet = NetworkFactory
    # NetworkFactory inherited from kp
    # kp.tl_cnvs is the output of make_tl_layer(256)
    # tl_pool(256)
    # go to pool(256,TopPool,LeftPool)
    # go to TopPool defined in py_utils/cpools/__init__.py
    # go to TopPoolFunction in py_utils/cpools/__init__.py
    # apply means x is the input of this layer

class BottomPool(nn.Module):
    def forward(self, x):
        return BottomPoolFunction.apply(x)

class LeftPool(nn.Module):
    def forward(self, x):
        return LeftPoolFunction.apply(x)

class RightPool(nn.Module):
    def forward(self, x):
        return RightPoolFunction.apply(x)

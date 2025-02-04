import torch
import torch.nn as nn

from .py_utils import kp, AELoss, _neg_loss, convolution, residual
from .py_utils import TopPool, BottomPool, LeftPool, RightPool

class pool(nn.Module):
    # first call
    # train.py :      nnet = NetworkFactory
    # NetworkFactory inherited from kp
    # kp.tl_cnvs is the output of make_tl_layer(256)
    # tl_pool(256)
    # go to pool(256,TopPool,LeftPool) in CenterNet-104.py
    # go to TopPool  defined in in py_utils/cpools/__init__.py
    # go to TopPool
    def __init__(self, dim, pool1, pool2):
        # first called pool1 is top_pool to make the heatmap
        # become a vector show that where is the largest value in each col
        # pool2 is a leftpool , it count from the right to the left to see where is the largest value in each row
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = self.pool1(P1_look_conv)

        # x--> conv+bn+relu-->leftpool to see largest in row---->
        #                                                         + ------>conv ----> toppool
        #  --> conv+bn+relu------------------------------------->

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = self.pool2(P2_look_conv)

        # x--> conv+bn+relu-->toppool to see largest in row---->
        #                                                         + ------>conv ----> leftpool
        #  --> conv+bn+relu------------------------------------->

        # the difference between pool 1 and pool 2 is the order apply toppool or leftpool first

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)
        # x--> conv+bn+relu-->leftpool to see largest in row---->
        #                                                         + ------>conv ----> toppool--
        #  --> conv+bn+relu------------------------------------->                              |
        #                                                                                      + --> conv+bn---+ relu-->conv+bn+relu
        #  --> conv+bn+relu-->toppool to see largest in row---->                               |               |
        #                                                         + ------>conv ----> leftpool--               |
        #  --> conv+bn+relu------------------------------------->                                              |
        #                                                                                                      |
        #  --> conv+bn-----------------------------------------------------------------------------------------


        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class pool_cross(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)
        pool1    = self.pool3(pool1)


        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)
        pool2    = self.pool4(pool2)
        # x--> conv+bn+relu--> top_pool-->BottomPool--
        #                                             +-->conv+bn----
        #  --> conv+bn+relu--> leftpool-->rightpool---               +--->relu---->conv+bn+relu
        #  --> conv+bn-----------------------------------------------

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)
# first call
# train.py :      nnet = NetworkFactory
# NetworkFactory inherited from kp
# kp.tl_cnvs is the output of make_tl_layer(256)
# tl_pool(256) in CenterNet-104.py
# go to pool(256,TopPool,LeftPool) in in CenterNet-104.py

        # x--> conv+bn+relu-->leftpool to see largest in row---->
        #                                                         + ------>conv ----> toppool--
        #  --> conv+bn+relu------------------------------------->                              |
        #                                                                                      + --> conv+bn---+ relu-->conv+bn+relu
        #  --> conv+bn+relu-->toppool to see largest in row---->                               |               |
        #                                                         + ------>conv ----> leftpool--               |
        #  --> conv+bn+relu------------------------------------->                                              |
        #                                                                                                      |
        #  --> conv+bn-----------------------------------------------------------------------------------------


class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)

class center_pool(pool_cross):
    def __init__(self, dim):
        super(center_pool, self).__init__(dim, TopPool, LeftPool, BottomPool, RightPool)

def make_tl_layer(dim):
    return tl_pool(dim)
# first call
# train.py :      nnet = NetworkFactory
# NetworkFactory inherited from kp
# kp.tl_cnvs is the output of make_tl_layer(256)
# tl_pool(256) in CenterNet-104.py

def make_br_layer(dim):
    return br_pool(dim)

def make_ct_layer(dim):
    return center_pool(dim)

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)
# first call
# train.py :      nnet = NetworkFactory
# NetworkFactory inherited from kp
# kp contains two kp_module part
# kp_module.low1 is the output of make_hg_layer() function
# args: 3,256,256,2,residual
# so the output is just two residual blocks with kernel 3,and input, output channels are 256

class model(kp):
    def __init__(self, db):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 80

        super(model, self).__init__(
            db, n, 2, dims, modules, out_dim,
            make_tl_layer=make_tl_layer,
            make_br_layer=make_br_layer,
            make_ct_layer=make_ct_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)

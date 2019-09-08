import pdb
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer
# . means the directory of kp.py file
# .kp_utils means the kp_utils.py file in the same directory with kp.py

class kp_module(nn.Module):
    # this module is call in class kp
    # the arguments:
    # kp_module(
    #                 n, dims, modules, layer=kp_layer,
    #                 # n =5
    #                 # dims    = [256, 256, 384, 384, 384, 512]
    #                 # modules = [2, 2, 2, 2, 2, 4]
    #                 # kp_layer is resisual as default argument
    #                 make_up_layer=make_up_layer,# from .utils import make_layer
    #                 make_low_layer=make_low_layer,# from .utils import make_layer
    #                 make_hg_layer=make_hg_layer,# default is  .utils import make_layer
    #                                             # but CenterNet-104.py need CenterNet-104.py make_hg_layer()
    #                 make_hg_layer_revr=make_hg_layer_revr,# from .utils import make_layer_revr
    #                 make_pool_layer=make_pool_layer, # from kp_utils import make_pool_layer
    #                 make_unpool_layer=make_unpool_layer, # from kp_utils import make_unpool_layer
    #                 make_merge_layer=make_merge_layer # from kp_utils import make_merge_layer
    #             )
    # since kp_module is recursive, # dims    = [256, 256, 384, 384, 384, 512]
    #     #                 # modules = [2, 2, 2, 2, 2, 4]
    # so for the first kp_module,residual block are all at 256 in_channels
    #            sencond                                from 256 input to 384 output channels
    #            third，fourth                          input output channels are both 384
    #            fifth                                  from input 384 to output 512
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        # modules = [2, 2, 2, 2, 2, 4]
        curr_mod = modules[0] # 2
        next_mod = modules[1] # 2

        # dims    = [256, 256, 384, 384, 384, 512]
        curr_dim = dims[0]  # 256
        next_dim = dims[1]  # 256

        # train.py :      nnet = NetworkFactory
        # NetworkFactory inherited from kp
        # kp contains two kp_module part
        #
        self.up1  = make_up_layer(# from .utils import make_layer
            3, curr_dim, curr_dim, curr_mod,
            #  256       256       2
            layer=layer, **kwargs
            #  residual
        )  #a nn.Sequential with two residual blocks whose kernel-sizes are (3,3) and input,output channels are 256

        self.max1 = make_pool_layer(curr_dim)
        # # from kp_utils import make_pool_layer, 256
        # has nothing to do with dim
        # this self.max1 is just a nn.MaxPool2d(kernel_size=2,stride=2) layer



        self.low1 = make_hg_layer(# CenterNet-104.py make_hg_layer()
            3, curr_dim, next_dim, curr_mod,
            #  256       256       2
            layer=layer, **kwargs
            #  residual
        )# two residual blocks with kernel 3,and input, output channels are 256

        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            # # from .utils import make_layer
            3, next_dim, next_dim, next_mod,
            #   256         256     2
            layer=layer, **kwargs
            # resisual
        )
        # n = 5 so for n = 4, n = 3, n = 2 kp_module will have recursive kp_modules as kp_module.low2
        # for n = 1  kp_module.low2 is just two residual blocks with kernel 3,and input, output channels are 256

        self.low3 = make_hg_layer_revr(
            # # from .utils import make_layer_revr
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )# so the output of make_layer_revr is just two residual blocks with the kernel size of 3, input , output channels 256
        self.up2  = make_unpool_layer(curr_dim)
        # # from kp_utils import make_unpool_layer  , 256
        # has nothing to do with dim, make_unpool_layer is just a nn.Upsample(scale_factor = 2)

        self.merge = make_merge_layer(curr_dim)
        # # from kp_utils import make_merge_layer
        # it is just a function to add two tensors together

    def forward(self, x):
        up1  = self.up1(x) # two residual blocks channel must stay the same since use same channel : curr_dim and curr_dim
        max1 = self.max1(x) # downsample
        low1 = self.low1(max1) # two block resisual blocks, channels may be more, because use curr_dim and next_dim
        low2 = self.low2(low1) # recursive kp_modules
        low3 = self.low3(low2) # two residual blocks and the channels may become less ,because use next_dim and curr_dim
        up2  = self.up2(low3) # upsampling
        return self.merge(up1, up2) # add up

class kp(nn.Module):
    # nn.Module --> kp --> CenterNet-104.py class: model
    def __init__(
        self, db, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer, make_ct_layer=make_ct_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):

        # train() call NetworkFactory---> CenterNet-104.py class model --> kp
        #         n       = 5
        #         dims    = [256, 256, 384, 384, 384, 512]
        #         modules = [2, 2, 2, 2, 2, 4]
        #         out_dim = 80
        #             db, n, 2, dims, modules, out_dim,
        #             make_tl_layer=make_tl_layer, # CenterNet-104.py make_tl_layer not the one in kp_utils.py
        #             make_br_layer=make_br_layer, # CenterNet-104.py make_br_layer not the one in kp_utils.py
        #             make_ct_layer=make_ct_layer, # CenterNet-104.py make_ct_layer not the one in kp_utils.py
        #             make_pool_layer=make_pool_layer, # CenterNet-104.py make_pool_layer not the one in kp_utils.py
        #             make_hg_layer=make_hg_layer, # CenterNet-104.py make_hg_layer not the one in kp_utils.py
        #             kp_layer=residual, cnv_dim=256
        super(kp, self).__init__()

        self.nstack             = nstack  # 2
        self._decode            = _decode # a funtion in kp_utils.py
        self._db                = db
        self.K                  = self._db.configs["top_k"]  # 70 in CenterNet-104.json
        self.ae_threshold       = self._db.configs["ae_threshold"] # 0.5 in CenterNet-104.json
        self.kernel             = self._db.configs["nms_kernel"] # 0.5 in CenterNet-104.json
        self.input_size         = self._db.configs["input_size"][0] # [511,511] in CenterNet-104.json
        self.output_size        = self._db.configs["output_sizes"][0][0] # [128,128] in CenterNet-104.json

        curr_dim = dims[0] # 256
        #  dims = [256, 256, 384, 384, 384, 512]


        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            # convolution(k, inp_dim, out_dim, stride=1, with_bn=True)
            #             k ---kernel size = 7
            #             input dim  = 3
            #             output channel = 128
            # convolution  = conv + BN + relu
            residual(3, 128, 256, stride=2)
            # residual(k, inp_dim, out_dim, stride=1, with_bn=True)
            # residual = conv + bn + relu + conv + bn + add + relu
        ) if pre is None else pre# in the arguments pre is None
        # self.pre is a block of convBR + resblock

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                # n =5
                # dims    = [256, 256, 384, 384, 384, 512]
                # modules = [2, 2, 2, 2, 2, 4]
                # kp_layer is resisual as default argument
                make_up_layer=make_up_layer,# from .utils import make_layer
                make_low_layer=make_low_layer,# from .utils import make_layer
                make_hg_layer=make_hg_layer,# default is  .utils import make_layer
                                            # but CenterNet-104.py need CenterNet-104.py make_hg_layer()
                make_hg_layer_revr=make_hg_layer_revr,# from .utils import make_layer_revr
                make_pool_layer=make_pool_layer, # from kp_utils import make_pool_layer
                make_unpool_layer=make_unpool_layer, # from kp_utils import make_unpool_layer
                make_merge_layer=make_merge_layer # from kp_utils import make_merge_layer
            ) for _ in range(nstack)    # nstack = 2
        ])
        # self.kps is two blocks of kp_module

        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
            # nstack = 2
            # from .kp_utils import make_cnv_layer
            # make_cnv_layer is just conv+bn+relu
            # and nstack is 2
            # so two conv+bn+ relu blocks is self.cns
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
            # the make_tl_layer function in CenterNet-104.py
            # cnv_dim = 256
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
            # from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
        ])

        self.ct_cnvs = nn.ModuleList([
            make_ct_layer(cnv_dim) for _ in range(nstack)
            # from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.ct_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for tl_heat, br_heat, ct_heat in zip(self.tl_heats, self.br_heats, self.ct_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            ct_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.ct_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image      = xs[0]
        tl_inds    = xs[1]
        br_inds    = xs[2]
        ct_inds    = xs[3]

        inter      = self.pre(image)
        outs       = []

        layers = zip(
            self.kps,      self.cnvs,
            self.tl_cnvs,  self.br_cnvs, 
            self.ct_cnvs,  self.tl_heats, 
            self.br_heats, self.ct_heats,
            self.tl_tags,  self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_,  br_cnv_  = layer[2:4]
            ct_cnv_,  tl_heat_ = layer[4:6]
            br_heat_, ct_heat_ = layer[6:8]
            tl_tag_,  br_tag_  = layer[8:10]
            tl_regr_,  br_regr_ = layer[10:12]
            ct_regr_         = layer[12]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            ct_cnv = ct_cnv_(cnv)

            tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
            tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
            tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)

            outs += [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_regr, br_regr, ct_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)

        outs          = []

        layers = zip(
            self.kps,      self.cnvs,
            self.tl_cnvs,  self.br_cnvs,
            self.ct_cnvs,  self.tl_heats,
            self.br_heats, self.ct_heats,
            self.tl_tags,  self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_,  br_cnv_  = layer[2:4]
            ct_cnv_,  tl_heat_ = layer[4:6]
            br_heat_, ct_heat_ = layer[6:8]
            tl_tag_,  br_tag_  = layer[8:10]
            tl_regr_,  br_regr_ = layer[10:12]
            ct_regr_         = layer[12]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                ct_cnv = ct_cnv_(cnv)

                tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
                tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
                         ct_heat, ct_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
                
        return self._decode(*outs[-8:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 8

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        ct_heats = outs[2::stride]
        tl_tags  = outs[3::stride]
        br_tags  = outs[4::stride]
        tl_regrs = outs[5::stride]
        br_regrs = outs[6::stride]
        ct_regrs = outs[7::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_ct_heat = targets[2]
        gt_mask    = targets[3]
        gt_tl_regr = targets[4]
        gt_br_regr = targets[5]
        gt_ct_regr = targets[6]
        
        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        ct_heats = [_sigmoid(c) for c in ct_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr, ct_regr in zip(tl_regrs, br_regrs, ct_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), (pull_loss / len(tl_heats)).unsqueeze(0), (push_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)

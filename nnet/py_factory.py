import os
import pdb
import torch
import importlib
import torch.nn as nn

from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss_kp  = self.loss(preds, ys, **kwargs)
        return loss_kp

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, db):
        # db is a MSCOCO instance.
        # and images, tl, br, center heatmaps and tl,br,center regression are all
        # prepared into torch.Tensor
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        # snapshot_name check CenterNet-104.json
        # snapshot = 5000  but snapshot_name wasn't in CenterNet-104.json
        # snapshot_name in config.py  is None
        # snapshot_name is set to "CenterNet-104" in train.py
        # so here module_file = models.CenterNet-104
        # it is to say we should go to models/CenterNet-104.py to see the classes

        print("module_file: {}".format(module_file))
        nnet_module = importlib.import_module(module_file)
        # this is importing the models/CenterNet-104.py
        # include five functions and 6 classes

        self.model   = DummyModule(nnet_module.model(db))
        # nnet_module.model(db) means CenterNet-104.py class model()
        # model inherit kp
        #
        self.loss    = nnet_module.loss
        # CenterNet-104.py  loss
        # loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)
        # AELoss in kp.py

        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes).cuda()

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, xs, ys, **kwargs):
        xs = [x for x in xs]
        ys = [y for y in ys]

        self.optimizer.zero_grad()
        loss_kp = self.network(xs, ys)
        loss        = loss_kp[0]
        focal_loss  = loss_kp[1]
        pull_loss   = loss_kp[2]
        push_loss   = loss_kp[3]
        regr_loss   = loss_kp[4]
        loss        = loss.mean()
        focal_loss  = focal_loss.mean()
        pull_loss   = pull_loss.mean()
        push_loss   = push_loss.mean()
        regr_loss   = regr_loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss, focal_loss, pull_loss, push_loss, regr_loss

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss_kp = self.network(xs, ys)
            loss       = loss_kp[0]
            focal_loss = loss_kp[1]
            pull_loss  = loss_kp[2]
            push_loss  = loss_kp[3]
            regr_loss  = loss_kp[4]
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f) 

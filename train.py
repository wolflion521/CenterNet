#!/usr/bin/env python
import os

import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CenterNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)

    #args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
    return args

def prefetch_data(db, queue, sample_data, data_aug):
    # train.py--> train()--->init_parallel_jobs --->for each thread: prefetch_data
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e
    # first call :training_dbs, training_queue, sample_data, True
    # train ---> init_paralel---> tasks call process --->for each process give a MSCOCO instance--->
    # so this method is the method called by one thread, not many thread.from
    # this method call kp_detection method, and return images,heatmaps,regressions...
    # below is how everything is returned.
    # data is a dictionary
    # return {
    #         "xs": [images, tl_tags, br_tags, ct_tags],# imagese is simple , other three are of shape (batch_size, max_tag_len)
                                                        # tl_tags, br_tags, ct_tags represent the location of keypoints in flattened image
    #         "ys": [tl_heatmaps, br_heatmaps, ct_heatmaps, tag_masks, tl_regrs, br_regrs, ct_regrs]
                                                        # tl_heatmaps, br_heatmaps, ct_heatmaps  3 heatmaps
                                                        # are of shape (batch_size,class_num,height,width)
                                                        # tl_regrs, br_regrs, ct_regrs are of shape (batch_size,max_tag_len,2)
    #     }, k_ind.

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn, data_aug):
    # # train.py--> train()--->init_parallel_jobs
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks
    # first this method is called by training task
    # training_dbs, training_queue, sample_data, True
    # training_dbs is a list of four MSCOCO instance, and MSCOCO instance is used for loading annotation data
    # the training_queue is a queue of 6
    # sample_data is a function


def train(training_dbs, validation_db, start_iter=0):
    # train.py--> here
    # training_db = [MSCOCO x 4] and use the dataset specified by "trainval2014"
    # validation_db is a MSCOCO instance whose configs should firstly check in file CenterNet-104.json
    # start_iter should check args.star_iter it should be 0

    learning_rate    = system_configs.learning_rate  # 0.00025
    max_iteration    = system_configs.max_iter       # 480000
    pretrained_model = system_configs.pretrain       # None
    snapshot         = system_configs.snapshot       # 5000
    val_iter         = system_configs.val_iter       # 500
    display          = system_configs.display        # 5
    decay_rate       = system_configs.decay_rate     # 10
    stepsize         = system_configs.stepsize       # 450000
    # all above hyperparameters should first check CenterNet-104.py to see ,
    # then check config.py

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size)
    # prefetch_size = 6  you can find this number in CenterNet.json
    validation_queue = Queue(5)
    #

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data)
    # sample.coco
    # and there is a coco.py in directory sample

    sample_data = importlib.import_module(data_file).sample_data
    # importlib.import_module(data_file)  means to import the sample.coco.py
    # and there is a function named sample_data in that file.
    # so sample_data means the function in sample/coco.py


    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data, True)
    # training_dbs is a list of four MSCOCO instance, and MSCOCO instance is used for loading annotation data
    # the training_queue is a queue of 6
    # sample_data is a function
    # four thread each thread load a batch of data.images, heatmaps,  location in flattened image, fractions part of keypoints

    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)


    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(training_dbs[0])

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss, focal_loss, pull_loss, push_loss, regr_loss = nnet.train(**training)
            #training_loss, focal_loss, pull_loss, push_loss, regr_loss, cls_loss = nnet.train(**training)

            if display and iteration % display == 0:
                print("training loss at iteration {}: {}".format(iteration, training_loss.item()))
                print("focal loss at iteration {}:    {}".format(iteration, focal_loss.item()))
                print("pull loss at iteration {}:     {}".format(iteration, pull_loss.item())) 
                print("push loss at iteration {}:     {}".format(iteration, push_loss.item()))
                print("regr loss at iteration {}:     {}".format(iteration, regr_loss.item()))
                #print("cls loss at iteration {}:      {}\n".format(iteration, cls_loss.item()))

            del training_loss, focal_loss, pull_loss, push_loss, regr_loss#, cls_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
                nnet.train_mode()

            if iteration % snapshot == 0:
                nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    # config is a class in config.py.
    # and system_configs is an instance of config
    # when system_config is initialized self._configs["config_dir"] = "config"

    # system_configs.config_dir ---------  config
    # so cfg_file  -------------------------  "config/CenterNet-104.json"
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    # configs datatype is dict, content can be checked in the file CenterNet-104.json
    # two blocks system and db
    # system block is about training process
    # db is about input size outputsize and augmentation
            
    configs["system"]["snapshot_name"] = args.cfg_file
    # args.cfg_file = "CenterNet-104"
    #
    system_configs.update_config(configs["system"])
    # before this line, parameters are set in the variable configs
    # configs is loaded from json file and change only one parameter
    # then we gonna put everything into system_configs, so we use update_config
    # and purpose is we will never use configs anymore, instead we will use system_configs
    # but now system_configs just read in the system part
    # the db part still need to update later from configs,
    # so configs can show out later

    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    # since system_configs most parameter is configed as the config.py illustrated,
    # and above line codes load the parameters from file CenterNet-104.json
    # so later on we will check CenterNet-104.json to figure out the settings
    #         "train_split": "trainval",
    #         "val_split": "minival",

    print("loading all datasets...")
    dataset = system_configs.dataset
    # in config.py file: self._configs["dataset"]           = None
    # in CenterNet-104.py file:    "dataset": "MSCOCO",
    # so here dataset = "MSCOCO"



    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads
    # parameters from args, go to --def argparser -- to check
    # here thread is 4

    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    # datasets is imported from db.datasets  that is a datasets.py file
    # and in that file datasets.py, there is only a dictionary----datasets
    # because    dataset="MSCOCO"
    # so         datasets[dataset] = datasets["MSCOCO"]
    # and        "MSCOCO" is a key in dictionary datasets, the value is MSCOCO
    # keep on    MSCOCO is imported from db.coco
    #            MSCOCO is a class inherited from DETECTION
    #            MSCOCO has   methods of :
    #                         load_data  load_coco_data  class_name   _extract_data    to_float
    #                         detection  evaluate        convert_to_coco

    # configs["db"]  should check the content in file CenterNet-104.json
    # below is configs["db"] content
    # "db": {
    #         "rand_scale_min": 0.6,
    #         "rand_scale_max": 1.4,
    #         "rand_scale_step": 0.1,
    #         "rand_scales": null,
    #
    #         "rand_crop": true,
    #         "rand_color": true,
    #
    #         "border": 128,
    #         "gaussian_bump": true,
    #
    #         "input_size": [511, 511],
    #         "output_sizes": [[128, 128]],
    #
    #         "test_scales": [1],
    #
    #         "top_k": 70,
    #         "categories": 80,
    #         "kp_categories": 1,
    #         "ae_threshold": 0.5,
    #         "nms_threshold": 0.5,
    #
    #         "max_per_image": 100
    #         }
    # all these db configs from json file input to MSCOCO class to initialize a MSCOCO instance
    #
    # MSCOCO inherited from DETECTION
    # DETECTION   has a dictionary named _config there are 28 key_value pairs,
    #
    # DETECTION is inherited from Base
    # in Base _config is an empty dictionary, 28 key_value pairs are set in db/detection.py
    # and MSCOCO class read in the dbconfigs( which we load from CenterNet-104.json )
    # and the parameters in file CenterNet-104.json will replace the values clarified in detection.py
    # so later on if we want to know the value of MSCOCO._config["key"],
    # we should first head on to CenterNet-104.json if the key cant find there,
    # we go to detection.py to check the value

    # finally training_dbs variable is a list of 4 MSCOCO instances
    # and here train_split = "trainval"
    # then check the init method in MSCOCO, we know that MSCOCO._dataset = "trainval2014"
    # finally:
    # training_db = [MSCOCO x 4] and use the dataset specified by "trainval2014"



    validation_db = datasets[dataset](configs["db"], val_split)
    # validation_db is a MSCOCO instance whose configs should firstly check in file CenterNet-104.json
    # and val_split = "minival", so here validation_db._dataset = "minival2014"

    print("system config...")
    pprint.pprint(system_configs.full)
    # this will print all configs
    # include
    #        training process: lr, epochs, decay, batchsize, network structure, pretrain....
    #        dataset:        image dir,   dataset configs...
    #        random strategy
    #        ....

    print("db config...")
    pprint.pprint(training_dbs[0].configs)
    # since training_dbs is a list of four MSCOCO instance,
    # here we just need the [0] ele in the list to represent the configs
    # remember these configs should first check CenterNet-104.py  then check detection.py

    print("len of db: {}".format(len(training_dbs[0].db_inds)))
    # training_dbs[0] is a MSCOCO instance,
    # so I check  MSCOCO._db_inds  I see: self._db_inds = np.arange(len(self._image_ids))
    #
    train(training_dbs, validation_db, args.start_iter)
    # training_db = [MSCOCO x 4] and use the dataset specified by "trainval2014"
    # validation_db is a MSCOCO instance whose configs should firstly check in file CenterNet-104.json

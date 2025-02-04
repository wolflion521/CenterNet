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
    # this function move the data into GPU and read them one batch after another each time
    # say the arguments are: training_queue, pinned_training_queue, training_pin_semaphore
    #     # training_queue is images, heatmaps,location regressions
    #     # pinned_training_queue hasn't initialized yet, it may be initialized in later line using function pin_memory
    #     # training_pin_semaphore is a counter
    # so data_queue structure can be checked in kp_detection()
    # data is a dictionary like below
    #     {
    #         "xs": [images, tl_tags, br_tags, ct_tags],
    #         "ys": [tl_heatmaps, br_heatmaps, ct_heatmaps, tag_masks, tl_regrs, br_regrs, ct_regrs]
    #     }
    # pinned_data_queue are not initialized
    # sema is a counter -- semaphore
    while True:
        data = data_queue.get()
        # data is a dictionary like below
        #     {
        #         "xs": [images, tl_tags, br_tags, ct_tags],
        #         "ys": [tl_heatmaps, br_heatmaps, ct_heatmaps, tag_masks, tl_regrs, br_regrs, ct_regrs]
        #     }
        # and all the elements in data are torch tensors.

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]
        # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
        #
        # 主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。
        #
        # 而显卡中的显存全部是锁页内存！
        #
        # 当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。
        # ————————————————
        # 版权声明：本文为CSDN博主「tsq292978891」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
        # 原文链接：https://blog.csdn.net/tsq292978891/article/details/80454568
        # tensor.pin_memory(): Copies the tensor to pinned memory, if it’s not already pinned.

        pinned_data_queue.put(data)
        # till this line I understand, dataqueue is reading and manipulating data in
        # CPU, and this self defined pin_memory function is used to copy the CPU data into GPU memory
        # pinned_data_queue share the same structure with data:
        # data is a dictionary like below
        #     {
        #         "xs": [images, tl_tags, br_tags, ct_tags],
        #         "ys": [tl_heatmaps, br_heatmaps, ct_heatmaps, tag_masks, tl_regrs, br_regrs, ct_regrs]
        #     }


        if sema.acquire(blocking=False):
            # acquire()  semaphore minus one ; release() semaphore plus one
            # 每调用一次acquire()，计数器减1；每调用一次release()，计数器加1.当计数器为0时，acquire()调用被阻塞。
            # Acquire a semaphore.
            # When invoked without arguments: if the internal counter is larger than zero on entry,
            # decrement it by one and return immediately. If it is zero on entry, block, waiting until
            # some other thread has called release() to make it larger than zero. This is done with
            # proper interlocking so that if multiple acquire() calls are blocked, release() will
            # wake exactly one of them up. The implementation may pick one at random, so the order
            # in which blocked threads are awakened should not be relied on. There is no return
            # value in this case.
            # When invoked with blocking set to false, do not block.
            # If a call without an argument would block, return false immediately;
            # otherwise, do the same thing as when called without arguments, and return true.
            return
        # since this semophore is inside a while loop whose task is reading training data. so we can
        # see this loop as a batch data

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
    # Process usage:   p = Process(target = function, args = (arguments input to function))
    # so here db is MSCOCO instance , queue is for restore the output of prefetch_data function.
    # after after the Process start(), annotation data and images are stored in queue.


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
    # Queue is for torch.multiprocessing module to share data, manipulate data, exchange data, operate data.
    # since torch.multiprocessing function can't return value. so the operation is using Queue

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)
    # queue.Queue is for threading to share data

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
    # training_tasks is a list of torch.multiprocessing.Process objects,
    # so when each Process object.start()   the original images and annotation files will be processed into
    # the formula accord with input shape to the network
    # data for training will be stored in training_queue

    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)
        # data for validation will be stored in validation_queue.


    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    # class threading.Semaphore([value])
    # values是一个内部计数，values默认是1，如果小于0，则会抛出 ValueError 异常，可以用于控制线程数并发数
    # here semaphore use default value so is 1
    # Semaphore 是 Python 内置模块 threading 中的一个类
    # Semaphore 管理一个计数器，每调用一次 acquire() 方法，
    # 计数器就减一，每调用一次 release() 方法，计数器就加一。
    # 计时器的值默认为 1 ，计数器的值不能小于 0，
    # 当计数器的值为 0 时，调用 acquire() 的线程就会等待，直到 release() 被调用。
    # 因此，可以利用这个特性来控制线程数量
    # 代码示例。
    # from threading import Thread, Semaphore
    # import time
    #
    #
    # def test(a):
    #     #打印线程的名字
    #     print(t.name)
    #     print(a)
    #     time.sleep(2)
    #     #释放 semaphore
    #     sem.release()
    #
    # #设置计数器的值为 5
    # sem = Semaphore(5)
    # for i in range(10):
    #     #获取一个 semaphore
    #     sem.acquire()
    #     t = Thread(target=test, args=(i, ))
    #     t.start()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    # training_queue is images, heatmaps,location regressions
    # pinned_training_queue hasn't initialized yet, it may be initialized in later line using function pin_memory
    # training_pin_semaphore is a counter
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    # Python Thread类表示在单独的控制线程中运行的活动。有两种方法可以指定这种活动：
    # 给构造函数传递回调对象：
    # https://blog.csdn.net/drdairen/article/details/60962439
    # target is a function
    # args is inputs for the function
    # the function pin_memory move the data into GPU and read them one batch after another each time
    training_pin_thread.daemon = True
    # daemon的使用场景是：你需要一个始终运行的进程，用来监控其他服务的运行情况，
    # 或者发送心跳包或者类似的东西，你创建了这个进程都就不用管它了，
    # 他会随着主线程的退出出而退出了。
    # so this line make the thread of reading data from CPU to GPU and read them batch by batch always alive
    # in the whole training stage
    training_pin_thread.start()
    #

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()
    # above four lines move validation data from CPU to GPU and read in batch by batch

    print("building model...")
    nnet = NetworkFactory(training_dbs[0])

    if pretrained_model is not None:# the CenterNet-104.json set pretrained model to None so these code skip
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter: # start_iter is 0
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)# set the learning rate to 0.00025

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

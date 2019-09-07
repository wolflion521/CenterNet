import cv2
import math
import numpy as np
import torch
import random
import string

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop, draw_gaussian, gaussian_radius

def _full_image_crop(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size   = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    return image, detections

def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def kp_detection(db, k_ind, data_aug, debug):
    ################################################################
    # kp_detectin , input whole dataset,
    # from dataset load a batch images and annotations
    # based on the annotations build relevant heatmat, regression tag,
    ################################################################

    # train.py--> train()--->init_parallel_jobs --->for each thread: prefetch_data---> sample_data ---> kp_detection
    # input: in training   db is a MSCOCO instance and dataset is trainval2014
    #        in validation db is a MSCOCO instance and dataset is minival2014
    # k_ind first call it is 0, then it will change inside kp_detection method. yes it is k_ind = (k_ind+1)%db_size
    # data_aug   is true when training , and it is false when validating
    # debug is set in sample_data method. it is set to False in both case
    data_rng   = system_configs.data_rng
    # check in config.py  data_rng = np.random.RandomState(123)
    batch_size = system_configs.batch_size
    # check in CenteNet-104.py   batch_size = 48

    # this is check in COCO class db_config content is listed below,
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
    # and above para is from CenterNet-104.py
    # if there is any para cant find in CenterNet-104,then goto db/detection.py to chekc

    categories   = db.configs["categories"]#  80
    input_size   = db.configs["input_size"]# [511,511]
    output_size  = db.configs["output_sizes"][0] # [ 128, 128]

    border        = db.configs["border"] # 128
    lighting      = db.configs["lighting"] # from detection.py   lighting  = true
    rand_crop     = db.configs["rand_crop"] # true
    rand_color    = db.configs["rand_color"] # true
    rand_scales   = db.configs["rand_scales"]
    # check CenterNet-104.json
    #         "rand_scale_min": 0.6,
    #         "rand_scale_max": 1.4,
    #         "rand_scale_step": 0.1,
    #         "rand_scales": null,
    # and check detection.py
    #             if self._configs["rand_scales"] is None:
    #             self._configs["rand_scales"] = np.arange(
    #                 self._configs["rand_scale_min"],
    #                 self._configs["rand_scale_max"],
    #                 self._configs["rand_scale_step"]
    #             )
    # so here rand_scales = np.arange(0.6,1.4,0.1) that is 0.6  0.7  0.8  0.9 ....  1.4


    gaussian_bump = db.configs["gaussian_bump"] # from detection.py   true
    gaussian_iou  = db.configs["gaussian_iou"] # from detection.py   0.7
    gaussian_rad  = db.configs["gaussian_radius"] # from detection.py  -1

    max_tag_len = 128

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
                            #  48     ,3  ,    511,           511
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
                            #  48     ,     80    ,      128      ,      128
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
                            #  48     ,     80    ,      128      ,      128
    ct_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
                            #  48     ,     80    ,      128      ,      128
    tl_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
                            #  48     ,     128    , 2
    br_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
                            #  48     ,     128    , 2
    ct_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
                            #  48     ,     128    , 2
    tl_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    br_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    ct_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)
                            #    48   ,

    db_size = db.db_inds.size
    # back to db/coco.py to check db.db_inds
    # self._db_inds = np.arange(len(self._image_ids))
    # so here db_size means how many images does this dataset has. eg.10000 images then db_size = 10000
    for b_ind in range(batch_size): # iterate images one by one
        if not debug and k_ind == 0:
            db.shuffle_inds()
            # since when we call, we always set debug to False no matter it is training or validation
            # and k_ind only have one chance to be 0, that is when we first call ke_detection
            # this shuffle_inds() method is written in base.py

        db_ind = db.db_inds[k_ind]
        # db_inds are shuffled in the first iteration, then take the index useing k_ind
        k_ind  = (k_ind + 1) % db_size
        #

        # reading image
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        # reading detections
        detections = db.detections(db_ind)
        # db is a MSCOCO instance, and MSCOCO.detection is written in db/coco.py
        # in train.py , MSCOCO is initialized and MSCOCO._detections are filled with all annotations infomation.
        # here db.detections(db_ind)
        # db_ind is the id of an image
        # then use the id to get the annotation of that image
        # so here detections is the label infomation of a single image

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
            # image is cropped and detections(bounding box is changed at the same time)
        else:
            image, detections = _full_image_crop(image, detections)

        image, detections = _resize_image(image, detections, input_size)
        # resize image and detections to another shape at the same time.
        # And there is risk that the detections are not within the boundaries of the image.
        detections = _clip_detections(image, detections)
        # so here clip the detections keep you away from above metioned risk.
        # make all the detections within the boundaries


        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        #input size and output size can be found in CenterNet-104.json
        # input size = 511,511
        # output size = 128,128
        # so width_ratio = 511/128 = 3.9921875

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))
        # make image to be channel first

        for ind, detection in enumerate(detections):
            # all these operations are for one single image
            # since below code will apply scale to detections,
            # detections should be integers not within (0,1) range
            category = int(detection[-1]) - 1
            #category = 0

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            xct, yct = (detection[2] + detection[0])/2., (detection[3]+detection[1])/2.

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            xct = int(fxct)
            yct = int(fyct)

            if gaussian_bump: # CenterNet-104 set to true
                width  = detection[2] - detection[0]# original value
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio) # multiply ratio so it is  for output size
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:# -1 means auto calculate gaussian rad
                    # match CenterNet-104 setting
                    radius = gaussian_radius((height, width), gaussian_iou)
                    # gaussian_iou = 0.7
                    radius = max(0, int(radius)) # eg. if an obj bounding box is 50,80, then the radius is just 17 or so
                else:
                    radius = gaussian_rad

                draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
                draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius, delte = 5)
                # all three inputs are zeros with shape    48     ,     80    ,      128      ,      128
                # tl_heatmaps[b_ind, category] is 128 x 128
                # top left corner
                # bottom right corner
                # center corner each one has an heatmap
                # about the delte para , topleft and bottom right are both set to 6,
                # why center heatmap set it to 5?
                # in draw_gaussian: sigma=diameter / delte  so the bigger delte ,the smaller sigma, and the heatmap value
                # in that keypoint is higher,
                # here it set the heatmap value of center keypoint larger than two corner keypoints.
                # important****** the

            else:
                tl_heatmaps[b_ind, category, ytl, xtl] = 1
                br_heatmaps[b_ind, category, ybr, xbr] = 1
                ct_heatmaps[b_ind, category, yct, xct] = 1
                # if---else   if is using gaussian distribution,and else if use only one peak

            tag_ind                      = tag_lens[b_ind]
            # tag_lens is (batch_size,)
            # and b_ind is the image index within batch
            # tag_lens is used to store how many detections the image has.
            # you can confirm with 6 lines below
            tl_regrs[b_ind, tag_ind, :]  = [fxtl - xtl, fytl - ytl]
            br_regrs[b_ind, tag_ind, :]  = [fxbr - xbr, fybr - ybr]
            ct_regrs[b_ind, tag_ind, :]  = [fxct - xct, fyct - yct]
            # all the three regression varibles are 3 dementional.
            # (b_ind,tag_ind,2)
            # for example. in one batch we have 48 images,
            # for each image we have differnt numbers of detections, may be first image has 4 detections.
            # may be the second has 15 detections.
            # but when we forward the network,we need it to have stable shape.
            # so here is how these arrays are initialized.
            # ct_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
            tl_tags[b_ind, tag_ind]      = ytl * output_size[1] + xtl
            br_tags[b_ind, tag_ind]      = ybr * output_size[1] + xbr
            ct_tags[b_ind, tag_ind]      = yct * output_size[1] + xct
            # these 3 arrays are used together with above three arrays.
            # these 3 are used to store the integer part of the scale to outputsize detection
            # the above 3 variables are used to store the fractions.
            # ct_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
            tag_lens[b_ind]             += 1

    for b_ind in range(batch_size):
        # for image in batches
        tag_len = tag_lens[b_ind]# how many detections the image has
        tag_masks[b_ind, :tag_len] = 1
        # tag_masks first appears in the begining of this method.
        # tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8) this is how it is initialized


    images      = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    ct_heatmaps = torch.from_numpy(ct_heatmaps)
    tl_regrs    = torch.from_numpy(tl_regrs)
    br_regrs    = torch.from_numpy(br_regrs)
    ct_regrs    = torch.from_numpy(ct_regrs)
    tl_tags     = torch.from_numpy(tl_tags)
    br_tags     = torch.from_numpy(br_tags)
    ct_tags     = torch.from_numpy(ct_tags)
    tag_masks   = torch.from_numpy(tag_masks)

    return {
        "xs": [images, tl_tags, br_tags, ct_tags],
        "ys": [tl_heatmaps, br_heatmaps, ct_heatmaps, tag_masks, tl_regrs, br_regrs, ct_regrs]
    }, k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    # train.py--> train()--->init_parallel_jobs --->for each thread: prefetch_data---> sample_data
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
    # system_configs.sampling_function = "kp_detection"  checked in CenterNet-104.json
    # db is a MSCOCO instance,it will deal with annotations parsing
    # k_ind : first call it is 0, then it may update to 1,2,3,.... in globals
    # data_aug : this flag in train.py  train() for training it is set to True, for validation it is set to False
    # globals() Return the dictionary containing the current scope's global variables.
    # NB. since it return the global variables in dictionary type. that is to say global is a dictionary with four key_value
    # pairs. and we set the key  to kp_detection.
    # so this line is calling kp_detection and feed the input to kp_detection
    # and

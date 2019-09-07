import sys
sys.path.insert(0, "data/coco/PythonAPI/")

import os
import json
import numpy as np
import pickle

from tqdm import tqdm
from db.detection import DETECTION
from config import system_configs
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class MSCOCO(DETECTION):
    def __init__(self, db_config, split):
        # db_config content is listed below,
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
        super(MSCOCO, self).__init__(db_config)
        # check detection.py
        # the keys should firstly check the values in CenterNet-104.json
        # if CenterNet-104.json didn't specify , we should check it in detection.py to look for default values



        data_dir   = system_configs.data_dir
        # CenterNet-104.py says "data_dir": "../data"
        # so data_dir = "../data"

        result_dir = system_configs.result_dir
        # result_dir is not specified in file CenterNet-104.py
        # so I look it up in config.py     self._configs["result_dir"] = "results"
        # so result_dir = "result"
        # may be here will mkdir in current directory
        #



        cache_dir  = system_configs.cache_dir
        # cache_dir is not specified in CenterNet-104.py either,
        # look it up in config.py       self._configs["config_dir"] = "config"
        # so cache_dir = "config"
        # may be a folder named config will be created later

        self._split = split
        # when the MSCOCO is training_dbs, the split is "trainval"
        #                    validation_db, the split is "minival"

        self._dataset = {
            "trainval": "trainval2014",
            "minival": "minival2014",
            "testdev": "testdev2017"
        }[self._split]
        # self.dataset is just a string, not a complex dictionary
        # when training_dbs, self._dataset ="trainval2014"
        # when validation_db, self._dataset = "minival2014"
        
        self._coco_dir = os.path.join(data_dir, "coco")
        # so self._coco_dir = "../data/coco"

        self._label_dir  = os.path.join(self._coco_dir, "annotations")
        # so self._label_dir = "../data/coco/annotations"

        self._label_file = os.path.join(self._label_dir, "instances_{}.json")
        # so self._label_file = "../data/coco/annotations/instances_{}.json"

        self._label_file = self._label_file.format(self._dataset)
        # training_dbs:    self._label_file = "../data/coco/annotations/instances_trainval2014.json"
        # validation_db:   self._label_file = "../data/coco/annotations/instances_minival2014.json"

        self._image_dir  = os.path.join(self._coco_dir, "images", self._dataset)
        # training_dbs:      self._image_dir = "../data/coco/images/trainval2014"
        # validation_db:     self._image_dir = "../data/coco/images/minival2014"

        self._image_file = os.path.join(self._image_dir, "{}")
        # training_dbs:      self._image_file = "../data/coco/images/trainval2014{}"
        # validation_db:     self._image_file = "../data/coco/images/minival2014{}"

        self._data = "coco"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        # here self._cat_ids  is just a init value
        # and the _cat_ids could be replaced later
        # for example in _extract_data method: the method will parse an annotation file
        # to determine which cat_ids is in that annotation file ,
        # for cat_ids not in that annotation file, we wont keep it in self._cat_ids

        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        # self._classes is a dictionary
        # and what is interesting is that key and value is the same,
        #

        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }
        # self._coco_to_class_map is reverse to self._cat_ids

        # so I guess sometimes self._cat_ids may not be continuous integers like here
        # maybe self._cat_ids can be list of string names of classes
        # or the class name may be 1,3,43,2,5, and it is not 1,2,3,4,5,6...
        # and the situation or self._cat_ids is randomly arranged need self._classes and self._coco_ro_class_map

        self._cache_file = os.path.join(cache_dir, "coco_{}.pkl".format(self._dataset))
        # cache_dir = "config"
        # self._dataset = "trainval2014"     "minival2014"
        # so for training _dbs:   self._cache_file ="config/coco_trainval2014.pkl"
        #    for validation_bd:   self._cache_file = "config/coco_minival2014.pkl"


        self._load_data()
        # load the annotation file data into self._detections and self._image_ids

        self._db_inds = np.arange(len(self._image_ids))
        # since self._image_ids are  something like  039239884,8283848,2324543,234334
        # and not easy to iterate
        # so we give self._image_ids another index named self._db_inds
        # later we will iterate self._db_inds to iterate self._image_ids

        self._load_coco_data()
        # so this method is to set self._coco_eval_ids

        # self._load_data() is meant to load the data for training .
        #                 because self._detections and self._image_ids is loaded
        # self._load_coco_data() is mean to load the data for validation
        #                 because self._coco_eval_ids are loaded




    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        #    for training _dbs:   self._cache_file ="config/coco_trainval2014.pkl"
        #    for validation_bd:   self._cache_file = "config/coco_minival2014.pkl"
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            # self._extract_data method is to load specific infomation
            # from original annotation dictionary. And we only extract bboxing and categories out
            # of the whole annotations, and hold these infomation with a dictionary named self._detections
            # so if we dont extract like this, we can train too, but the data io will be mess.
            # so after this line, self._detections is ready, we can use it to train.

            # and why there is an if-else structure? because, self._extract_data is time wasting, we just want
            # to do it once, and after the self._detection variable is ready we want to save it into a pkl file
            # and later when we use it ,we just need to load the pkl


            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)
        # so self._load_data and self._extract_data are equvalent, they do the same thing
        # they all set the self._detections, and self._image_ids variables

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        # training_dbs:    self._label_file = "../data/coco/annotations/instances_trainval2014.json"
        # validation_db:   self._label_file = "../data/coco/annotations/instances_minival2014.json"
        # COCO is imported from pycocotools.coco
        # so self._coco now is a COCO instance with annotations
        with open(self._label_file, "r") as f:
            data = json.load(f)
        # so data is a complete annotation
        # with five parts:
        # {
        #     "info": info,
        #     "licenses": [license],
        #     "images": [image],
        #     "annotations": [annotation],
        #     "categories": [category]
        # }

        coco_ids = self._coco.getImgIds()
        # I guess this just get the images part from the dictionary above
        # and the ids returned by cocoinstance.getImgIds() can iterate each image

        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }
        # iterate each image to get the image_filename and id,
        # and make image filename and image id become a dictionary

        self._coco_categories = data["categories"]
        # "categories": [ # 类别描述
        #         {
        #             "supercategory": "person", # 主类别
        #             "id": 1, # 类对应的id （0 默认为背景）
        #             "name": "person" # 子类别
        #         },
        #         {
        #             "supercategory": "vehicle",
        #             "id": 2,
        #             "name": "bicycle"
        #         },
        #         {
        #             "supercategory": "vehicle",
        #             "id": 3,
        #             "name": "car"
        #         },
        #         ……
        #         ……
        #     ],
        self._coco_eval_ids   = eval_ids
        # so this method is to set self._coco_eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat    = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco    = COCO(self._label_file)
        # training_dbs:    self._label_file = "../data/coco/annotations/instances_trainval2014.json"
        # validation_db:   self._label_file = "../data/coco/annotations/instances_minival2014.json"
        # COCO is imported from pycocotools.coco
        # so self._coco now is a COCO instance with annotations

        # and here is an example of an coco annotation file
        # {
        #     "info": info, # dict
        #     "licenses": [license], # list ，内部是dict
        #     "images": [image], # list ，内部是dict
        #     "annotations": [annotation], # list ，内部是dict
        #     "categories": # list ，内部是dict
        # }


        self._cat_ids = self._coco.getCatIds()
        # here the self._cat_ids are no longer perfect 1,2,3,4
        # may be it is like   1,34,  2,   3,87  ....
        # so we need two dictionaries:
        # one is translate id to catgorical_number
        # the other is translating catogorical number back to ids
        # and these two dictionary is
        # self.class    ans    self._coco_to_class_map
        # so if there are only 40 classes annotations, then self._cat_ids will only keep these 40 numbers
        # here is an example of categories value
        # "categories": [ # 类别描述
        #         {
        #             "supercategory": "person", # 主类别
        #             "id": 1, # 类对应的id （0 默认为背景）
        #             "name": "person" # 子类别
        #         },
        #         {
        #             "supercategory": "vehicle",
        #             "id": 2,
        #             "name": "bicycle"
        #         },
        #         {
        #             "supercategory": "vehicle",
        #             "id": 3,
        #             "name": "car"
        #         },
        #         ……
        #         ……
        #     ],



        coco_image_ids = self._coco.getImgIds()
        # this is all the images ids maybe tens of thousands long
        # "images": [
        #         {
        #             "license": 4,
        #             "file_name": "000000397133.jpg",
        #             "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        #             "height": 427,
        #             "width": 640,
        #             "date_captured": "2013-11-14 17:02:52",
        #             "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        #             "id": 397133
        #         }
        #     ]
        # in annotation files images is a list
        # and images contain many images.
        # for each image it offer a dict, and file name is one key in that dictionary
        # look at following line , I can see that COCO_instance.lodaImgs(img_id)[0] will return a dictionary just like above

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"] 
            for img_id in coco_image_ids
        ]
        # here I learn the utilise of COCO
        # first COCO_instance.getImgIds     get   some kind of ID of all images
        # then use COCO_instace.loadImgs(ID)  for ID in IDs   to get a dictionary,
        # and if we want the img_file_name we need to get the value of key "file_name"
        # so here self._image_ids is a list of all img paths in "../data/coco/annotations/instances_trainval2014.json"
        # or "../data/coco/annotations/instances_minival2014.json"

        self._detections = {}


        # below is a pipeline to extract images,  there are 3 for loops
        # first one :  for each image
        #                  then in that image,for each categorical_ids:
        #                       then for each bbox in boxes:
        #                       and bbox and catogorical is stored in two numpy arrays named bboxes and categories
        #                   and the annotation numpy array will stack into a biggest dictionary named self._detections
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image      = self._coco.loadImgs(coco_image_id)[0]
            # here image is not a picture yet, it is just a dictionary
            # and image in fact is a dictionary
            # has a key named "file_name" which restore the address of image file

            bboxes     = []
            categories = []

            for cat_id in self._cat_ids:
                # self._cat_ids  is total class of this annotation file
                # so self._cat_ids is sth like 80 or 40   it is not 100000  or 20000

                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                # image is a dictionary .   for example
                #         #         {
                #         #             "license": 4,
                #         #             "file_name": "000000397133.jpg",
                #         #             "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
                #         #             "height": 427,
                #         #             "width": 640,
                #         #             "date_captured": "2013-11-14 17:02:52",
                #         #             "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
                #         #             "id": 397133
                #         #         }
                # so imgIds is 397133

                # here I guess getAnnIds will extract from the dictionary of just this image dictionary and
                # find all the candidates with the catogorical_ID specified by cat_id
                # so this line is used to gather all the pictures with same class object together


                annotations    = self._coco.loadAnns(annotation_ids)
                # above line gather same class candidates and here we read in the annotations of these candidates



                category       = self._coco_to_class_map[cat_id]
                # id become category name

                for annotation in annotations:
                    # in this for loop, each loop process an annotation
                    bbox = np.array(annotation["bbox"])
                    # extract bounding box from the annotation
                    # if someday I want to extract masks maybe I just need to write annotation["mask"]
                    bbox[[2, 3]] += bbox[[0, 1]]
                    # obviously bbox is left,top,width,height
                    # after this line bbox become left,top,right,bottom
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes     = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))
        # so finally the purpose of self._extract_data method is to load specific infomation
        # from original annotation dictionary. And we only extract bboxing and categories out
        # of the whole annotations, and hold these infomation with a dictionary named self._detections
        # so if we dont extract like this, we can train too, but the data io will be mess.

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids  = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval.evaluate_fd()
        coco_eval.accumulate_fd()
        coco_eval.summarize_fd()
        return coco_eval.stats[0], coco_eval.stats[12:]

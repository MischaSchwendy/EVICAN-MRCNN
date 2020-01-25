"""
Mask R-CNN
Configurations and data loading code for EVICAN.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Train on the EVICAN dataset by Schwendy et al.
https://edmond.mpdl.mpg.de/imeji/collection/l45s16atmi6Aa4sI

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla, adapted by Mischa Schwendy

------------------------------------------------------------

Usage: run from the command line as such:

    # Train a new model starting from pre-trained EVICAN weights
    python3 evican.py train --dataset=/path/to/evican/ --model=evican.h5

    # Continue training a model that you had trained earlier
    python3 evican.py train --dataset=/path/to/evican/ --model=/path/to/weights.h5

    # Run EVICAN evaluation on a model you trained
    # Lowest difficulty
    python3 evican.py evaluate --difficulty=_easy --dataset=/path/to/evican/ --model=/path/to/weights.h5

    # Medium difficulty
    python3 evican.py evaluate --difficulty=_medium --dataset=/path/to/evican/ --model=/path/to/weights.h5

    # Highest difficulty
    python3 evican.py evaluate --difficulty=_difficult --dataset=/path/to/evican/ --model=/path/to/weights.h5
"""

import os
import sys
import time
import numpy as np

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# Pycocotools2 are needed for EVICAN, as small adaptations allow evaluation solely on segmentations.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools2.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
EVICAN_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2019"

############################################################
#  Configurations
############################################################


class EvicanConfig(Config):
    """Configuration for training on EVICAN.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "evican"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # EVICAN has 2 classes


    use_multiprocessing=False
    MAX_GT_INSTANCES = 1

############################################################
#  Dataset
############################################################

class EvicanDataset(utils.Dataset):
    def load_evican(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_evican=False, difficulty=""):
        """Load a subset of the EVICAN dataset.
        dataset_dir: The root directory of the EVICAN dataset.
        subset: What to load (train, val, eval)
        year: What dataset year to load (so far only 2019) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        return_evican: If True, returns the EVICAN object.
        Difficulty: Level for evaluation (_easy, _medium, _difficult)
        """
        #create evican object
        evican = COCO("{}/annotations/instances_{}{}{}.json".format(dataset_dir, subset, year, difficulty))
        if subset == "train":
            dset = "train"
        elif subset == "val":
            dset = "val"
        elif subset == "eval":
            dset = "eval"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)


        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(evican.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(evican.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(evican.imgs.keys())

        # Add classes
        #self.add_class("evican", 1, "Cell")
        #self.add_class("evican", 2, "Nucleus")
        # Add classes
        for i in class_ids:
            self.add_class("evican", i, evican.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "evican", image_id=i,
                path=os.path.join(image_dir, evican.imgs[i]['file_name']),
                width=evican.imgs[i]["width"],
                height=evican.imgs[i]["height"],
                annotations=evican.loadAnns(evican.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_evican:
            return evican

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "evican.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(EvicanDataset, self).load_mask(image_id)


    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  EVICAN Evaluation
############################################################

def build_evican_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "evican"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_evican(model, dataset, evican, eval_type="segm", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "segm" for segmentation evaluation, as only segmentations were provided (no bounding boxes!)
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick EVICAN images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding EVICAN image IDs.
    evican_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_evican_results(dataset, evican_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    evican_results = evican.loadRes(results)

    # Evaluate
    evicanEval = COCOeval(evican, evican_results, eval_type)
    evicanEval.params.imgIds = evican_image_ids
    evicanEval.evaluate()
    evicanEval.accumulate()
    evicanEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on EVICAN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on EVICAN")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/evican/",
                        help='Directory of the EVICAN dataset')
    parser.add_argument('--difficulty', required=False,
                        default = '',
                        metavar="Difficulty level",
                        help='Difficulty level')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the EVICAN dataset (2019)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'evican'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=33,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=33)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Difficulty: ", args.difficulty)
    print("Year: ", args.year)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = EvicanConfig()
    else:
        class InferenceConfig(EvicanConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "evican":
        model_path = EVICAN_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = EvicanDataset()
        dataset_train.load_evican(args.dataset, "train", year=args.year)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = EvicanDataset()
        val_type = "val"
        dataset_val.load_evican(args.dataset, val_type, year=args.year)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads',
                    augmentation=None)
        # learning rate decreased to 50%
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 2,
                    epochs=20,
                    layers='heads',
                    augmentation=None)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=4,
                    layers='4+',
                    augmentation=None)
        # learning rate decreased to 50%
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 2,
                    epochs=4,
                    layers='4+',
                    augmentation=None)

        # Training - Stage 3
        # Fine tune all layers, learning rate decreased to 10 %
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=2,
                    layers='all',
                    augmentation=None)
        # learning rate decreased to 5%
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 20,
                    epochs=2,
                    layers='all',
                    augmentation=None)

    elif args.command == "evaluate":
        # Evaluation dataset
        dataset_eval = EvicanDataset()
        eval_type = "eval"
        evican = dataset_eval.load_evican(args.dataset, eval_type, year=args.year, return_evican=True, difficulty = args.difficulty)
        dataset_eval.prepare()
        print("Running EVICAN evaluation on {} images.".format(args.limit))
        evaluate_evican(model, dataset_eval, evican, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

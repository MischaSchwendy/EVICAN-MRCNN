# EVICAN-MRCNN
This is an extension to Matterport Inc’s implementation of Mask R-CNN [https://github.com/matterport/Mask_RCNN] using the EVICAN Dataset by Schwendy et al. [https://edmond.mpdl.mpg.de/imeji/collection/l45s16atmi6Aa4sI].
Installation: Follow the installation instructions under https://github.com/matterport/Mask_RCNN and copy pycocotools2 and evican.py to samples/coco. To train on the EVICAN dataset use following data structure:

  •	Download the files from https://edmond.mpdl.mpg.de/imeji/collection/l45s16atmi6Aa4sI
  •	Create a parent folder called “EVICAN”
  •	Inside this folder, create the subfolders “annotations”, “train2019”, “val2019”, and “eval2019” 
  •	“annotations” includes the JSON files with the segmentation information. Erase the version information at the end of the JSON filenames, i.e. EVICAN2 or EVICAN60. Names must follow this format:
    o	Instances_train2019.json (for the training document)
    o	Instances_val2019.json (for the validation document)
    o	Instances_eval2019_easy.json (for the easy evaluation document)
    o	Instances_eval2019_medium.json (for the medium evaluation document)
    o	Instances_eval2019_difficult.json (for the difficult evaluation document)
  •	“train2019” includes all training images
  •	“val2019” includes all validation images
  •	“eval2019” includes all evaluation images
  •	To train the classifier run “python3 evican.py train --dataset=/path/to/EVICAN –model=/path/to/model”
  •	An easy starting point for classifier training is the pretrained model mask_rcnn_coco.h5 available from https://github.com/matterport/Mask_RCNN/releases
  •	To evaluate the segmentation accuracy run “python3 evican.py evaluate -–dataset=/path/to/EVICAN –-difficulty=_difficulty” with _difficulty being _easy, _medium, or _difficult according to the difficulty of the dataset used.


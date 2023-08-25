"""eval_yolo.py
run : python3 eval_yolo.py -m ./Models/yolov4-tiny-832.trt
run :eval_yolo.py -m yolov4-tiny-832


This script is for evaluating mAP (accuracy) of YOLO models.
"""


import os
import sys
import json
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#from coco import COCO
# from pycocotools.cocoeval import COCOeval
#from cocoeval import COCOeval
from progressbar import progressbar

from yolo_with_trt_plugins import TrtYOLO
from yolo_classes import yolo_cls_to_ssd



#HOME = os.environ['home']
VAL_IMGS_DIR = 'train/'
VAL_ANNOTATIONS = 'Yolo-to-COCO-format-converter/output/train.json'


def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLO model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs_dir', type=str, default=VAL_IMGS_DIR,
        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument(
        '--annotations', type=str, default=VAL_ANNOTATIONS,
        help='groundtruth annotations [%s]' % VAL_ANNOTATIONS)
    parser.add_argument(
        '--non_coco', action='store_true',
        help='don\'t do coco class translation [False]')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny|yolov8s_trt|yolov8m_trt]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608, 640) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)


def generate_results(trt_yolo, imgs_dir, jpgs, results_file, non_coco):
    """Run detection on each jpg and write results to file."""
    results = []
    
    for jpg in progressbar(jpgs):
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        image_id = int(jpg.split('.')[0].split('/')[-1])
        #image_id = jpg.split('.')[0]
        #print(image_id)
        boxes, confs, clss = trt_yolo.detect(img, conf_th=1e-2)
        # dataset = json.load(open(VAL_ANNOTATIONS, 'r'))
        # ann=dataset['annotations']
        # print(image_id)
        # print("cls:",clss)
        for box, conf, cls in zip(boxes, confs, clss):
            x = float(box[0])
            y = float(box[1])
            #w = float(box[2])
            #h = float(box[3])

            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            cls = int(cls)
            cls = cls if non_coco else yolo_cls_to_ssd[cls]
            results.append({'image_id': image_id,
                            'category_id': cls,
                            'bbox': [x, y, w, h],
                            'score': float(conf)})
    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))

def main():
    args = parse_args()
    check_args(args)
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('./model/%s.engine' % args.model):

    #if not os.path.isfile(args.model):
        raise SystemExit('ERROR: file (./%s) not found!' % args.model)

    results_file = './model/results_%s.json' % args.model
    #a='yolov4-tiny-832'
    # args.model[-19:-4]
    #results_file = 'Models/results_%s.json' % args.model[-19:-4]
    trtpath='./model/yolov8s_trt.engine'
    #trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
    trt_yolo = TrtYOLO(trtpath, args.category_num, args.letter_box)
    #print("non coco::",args.non_coco)

    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    
    #jpgs=sorted(jpgs,key=lambda jpg:int(jpg.split('.')[0].split('_')[-1])) #changed
    generate_results(trt_yolo, args.imgs_dir, jpgs, results_file,
                     non_coco=True)

    # generate_results(trt_yolo, args.imgs_dir, jpgs, results_file,
    #                  non_coco=args.non_coco)

    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

    cocoGt = COCO(args.annotations) 
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
    #cocoEval.params.catIds = sorted(cocoGt.getCatIds())
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    #eval=cocoEval.eval
    #print("p useCats",p.useCats) # 1
    #print("p maxDets",p.maxDets)
    # gts=cocoGt.loadAnns
    # dts=cocoDt.loadAnns
    # print("gts:",gts)
    # print("dts",dts)

    
if __name__ == '__main__':
    main()

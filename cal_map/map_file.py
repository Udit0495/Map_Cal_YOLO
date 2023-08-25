from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load ground truth annotations
gt_coco = COCO('train.json')

# Load predicted bounding box results
pred_coco = gt_coco.loadRes('result_file_final.json')

# Create COCO evaluation object
coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')

# Run evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Get Average Precision (AP) and mean Average Precision (mAP) per class
ap_per_class = coco_eval.stats[:gt_coco.cats]
mAP = coco_eval.stats[0]

# Print AP and mAP per class
for cat_id, cat_name in gt_coco.cats.items():
    print('Category: {} - AP: {:.4f}'.format(cat_name['name'], ap_per_class[cat_id]))

# Print mean Average Precision (mAP)
print('mAP: {:.4f}'.format(mAP))


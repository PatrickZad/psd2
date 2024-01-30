import fiftyone as fo
import fiftyone.zoo as foz
 
dataset = foz.load_zoo_dataset("sama-coco", splits="train", label_types="detections", include_id=True)
print(dataset)
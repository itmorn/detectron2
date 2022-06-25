"""
@Auth: itmorn
@Date: 2022/6/25-0:04
@Email: 12567148@qq.com
"""
import os
import json

def save_labelme(predictions, out_filename):
    instances = predictions["instances"].to("cpu")
    map_label = {0:"name",1:"num"}
    imagePath = os.path.basename(out_filename)
    h,w = instances.image_size
    dic_all = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
        ],
        "imagePath": imagePath,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    for i in range(len(instances)):
        instance = instances[i]
        fields = instance._fields
        pred_boxes = fields['pred_boxes'].tensor.numpy().tolist()[0]
        pred_classes = fields['pred_classes'].numpy().tolist()[0]
        dic_inner = {
          "label": map_label[pred_classes],
          "points": [
            pred_boxes[:2],
            pred_boxes[-2:]
          ],
          "group_id": None,
          "shape_type": "rectangle",
          "flags": {}
        }
        dic_all["shapes"].append(dic_inner)

    line = json.dumps(dic_all,indent=4)
    f = open(out_filename.replace(".jpg",".json"),"w",encoding="utf-8")
    f.write(line)
    f.close()

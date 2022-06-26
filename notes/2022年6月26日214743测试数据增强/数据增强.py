"""
@Auth: itmorn
@Date: 2022/6/26-21:48
@Email: 12567148@qq.com
"""
import cv2
import numpy as np

from detectron2.data import transforms as T
# Define a sequence of augmentations:
augs = T.AugmentationList([
    T.RandomBrightness(0.9, 1.1),
    T.RandomFlip(prob=1),
    # T.RandomCrop("absolute", (640, 640))
])  # type: T.Augmentation

# Define the augmentation input ("image" required, others optional):
image =cv2.imread("1391003.jpg",0)
boxes = np.array([178,198,247,544])
input = T.AugInput(image, boxes=boxes)
# Apply the augmentation:
transform = augs(input)  # type: T.Transform
image_transformed = input.image  # new image
sem_seg_transformed = input.sem_seg  # new semantic segmentation

# For any extra data that needs to be augmented together, use transform, e.g.:
image2_transformed = transform.apply_image(image2)
polygons_transformed = transform.apply_polygons(polygons)
"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch
import json

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):

        with open(csv_file) as f:
            annotations = json.load(f)
            annotations = list(annotations.items())
            self.annotations = np.array(annotations)

        self.img_dir = img_dir
        # self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        current_ann = self.annotations[index]
        img_path = os.path.join(self.img_dir, current_ann[0])
        image = np.array(Image.open(img_path).convert("RGB"))
        img_width, img_height = image.shape[0], image.shape[1]

        bboxes = [] # current_ann[1:][0] #[]
        # print(bboxes)
        # bboxes.append(0)
        #print("before alb img w {} img h {}".format(img_width, img_height))

        for ind, box in enumerate(current_ann[1:][0]):
            # print("height comp")
            # print(box)

            # print(img_height)
            #
            if box[0] + box[2] >= img_width or box[1] + box[3] >= img_height:
               continue
            # temp_box = []
            # temp_box.append(box[0] / width)
            # temp_box.append(box[1] / height)
            # temp_box.append(box[2] / width)
            # temp_box.append(box[3] / height)

            box.append(1)
            bboxes.append(box)


        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        img_width, img_height = image.shape[1], image.shape[2]
        # print("after alb img w {} img h {}".format(img_width, img_height))
        #
        # print("\n\n")
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            # print("before x {} y {} width {} height {}".format(x, y, width, height))
            # print("img w {} img h {}".format(img_width, img_height))
            x /= img_width
            y /= img_height
            width /= img_width
            height /= img_height

            # print("x {} y {} width {} height {}".format(x, y, width, height))
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                # print("S", S)
                # print("x {} y {}".format(x, y))
                i, j = int(S * y), int(S * x)  # which cell

                # print("i, j ", i, j)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        r"C:\Users\m\Desktop\COCOtestset\data.json",
        r"C:\Users\m\Desktop\COCOtestset\new_val2017",
        # "COCO/labels/labels_new/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()

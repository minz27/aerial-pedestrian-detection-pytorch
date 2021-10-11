import torch.utils.data
import os
import pandas as pd
from PIL import Image

import utils_

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path: str = "", transforms: object = None, filter_images: bool = True) -> None:
        self.path = path
        self.images = os.listdir(os.path.join(self.path, "data", "generated", "images"))
        self.transforms = transforms
        # self.width, self.height = 480, 480

        self.classes = ["_", "Pedestrian", "Biker", "Cart", "Skater", "Car", "Bus"]

        if filter_images:
            self.images = [img for img in self.images if img.split("_")[0] in ["bookstore", "coupa"]]
            self.images = [img for idx, img in enumerate(self.images) if len(self.__getitem__(idx)[1]["boxes"]) > 0]
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> object:
        image_file = self.images[idx]
        image = Image.open(os.path.join(self.path, "data", "generated", "images", image_file)).convert("RGB")

        loc, video, frame = image_file.split("_")
        frame = frame.split(".jpg")[0]

        annotations = pd.read_csv(
            os.path.join(self.path, "data", "annotations", loc, video, "annotations.txt"),
            sep=" ",
            names=["TrackID", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
        )
        annotations = annotations[annotations["frame"] == int(frame)]
        boxes, labels = [], []
        for _, row in annotations.iterrows():

            if row["lost"] or row["occluded"]:
                continue

            box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            boxes += [box]
            labels += [self.classes.index(row["label"])]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            # "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor(0.),
            # "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            # "image_id": torch.tensor([idx])
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


if __name__ == "__main__":
    dataset = Dataset(filter_images=True)
    print(f"len: {len(dataset)}")

    image, target = dataset[205]
    utils_.plot_bboxes(image, target)
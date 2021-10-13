import cv2
import os
import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches

"""
Custom utilities (additionally to those in utils.py from https://github.com/pytorch/vision/tree/master/references/detection)
"""

def generate_images() -> None:
    for loc_dir in os.listdir(os.path.join("data", "video")):
        print(f"Location {loc_dir}...")
        for video_dir in os.listdir(os.path.join("data", "video", loc_dir)):
            print(f"Video {video_dir}...")
            cam = cv2.VideoCapture(os.path.join("data", "video", loc_dir, video_dir, "video.mp4"))
            current_frame = 0
            while True:
                ret, frame = cam.read()
                if ret:
                    if current_frame % 100 == 0:
                        cv2.imwrite(os.path.join("data", "generated", "images", f"{loc_dir}_{video_dir}_{str(current_frame).zfill(5)}.jpg"), frame)
                    current_frame += 1
                else:
                    break
            cam.release()
            cv2.destroyAllWindows()

def get_transform(train: bool):
    transforms = [T.ToTensor()]
    if train:
        transforms += [T.RandomHorizontalFlip(.5)]
    return T.Compose(transforms)

def plot_bboxes(image, target) -> None:
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(image)
    for box in (target["boxes"]):
        x, y, width, height  = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = matplotlib.patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="r", facecolor="none")
        a.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    generate_images()
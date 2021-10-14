# FlyAI

Fine-tuning a pre-trained object detection model for the [Standford Drone Dataset](https://www.kaggle.com/aryashah2k/stanford-drone-dataset).

![img](https://github.com/nicolaskolbenchlag/aerial-pedestrian-detection/images/bookstore_video0_10000.jpg)

Code in sdd-utils comes from [here](https://github.com/JosephKJ/SDD-Utils).

## Tensorflow-Lite Approach

### Requirements & Dataset extraction

- Requires [ffmpeg](https://ffmpeg.org/) to be installed.

```cmd
pip install -r requirements.txt
```

and for to create images from video frames and create Pascal VOC dataset.

```cmd
sdd-utils/annotate.py
```

## PyTorch Approach

```cmd
cd pytorch
```

### Requirements

```cmd
pip install -r requirements.txt
```

### Run

#### Generate images from videos

```cmd
python src/utils_.py
```

#### Train

```cmd
python src/main.py
```

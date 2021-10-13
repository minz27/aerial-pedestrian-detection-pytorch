import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith("2")

tf.get_logger().setLevel("ERROR")
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get("efficientdet_lite0")
data_loader = object_detector.DataLoader.from_pascal_voc(images_dir="data/sdd/JPEGImages", annotations_dir="data/sdd/Annotations", label_map=["Pedestrian", "Biker", "Cart", "Skater", "Car", "Bus"], max_num_images=10000)
train_data = data_loader

model = object_detector.create(train_data, model_spec=spec, epochs=2, batch_size=16, train_whole_model=True)

model.export(export_dir="models/tf-lite")
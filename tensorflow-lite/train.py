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
data_loader = object_detector.DataLoader.from_pascal_voc(images_dir="data/sdd/JPEGImages", annotations_dir="data/sdd/Annotations", label_map=["Pedestrian", "Biker", "Cart", "Skater", "Car", "Bus"])

train_data, tmp = data_loader.split(.6)
val_data, test_data = tmp.split(.5)

model = object_detector.create(train_data, model_spec=spec, epochs=50, batch_size=8, train_whole_model=True, validation_data=val_data)

model.evaluate(test_data)
model.export(export_dir="models/tf-lite")
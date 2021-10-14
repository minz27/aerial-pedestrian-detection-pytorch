from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith("2")

tf.get_logger().setLevel("ERROR")
from absl import logging
logging.set_verbosity(logging.ERROR)

def train():
    spec = model_spec.get("efficientdet_lite0")
    
    labels = ["Pedestrian", "Biker", "Cart", "Skater", "Car", "Bus"]

    data_loader = object_detector.DataLoader.from_pascal_voc(images_dir="data/sdd/JPEGImages", annotations_dir="data/sdd/Annotations", label_map=labels, max_num_images=None)

    model = object_detector.create(data_loader, model_spec=spec, epochs=1, batch_size=64, train_whole_model=False)
    model.export(export_dir="models/tf-lite-02")

if __name__ == "__main__":
    train()
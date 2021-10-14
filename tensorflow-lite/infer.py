import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "../models/model.tflite"

label_map = ["Pedestrian", "Biker", "Cart", "Skater", "Car", "Bus"]
classes = label_map

# classes = ['???'] * len(label_map)
# for label_id, label_name in label_map.as_dict().items():
#   classes[label_id-1] = label_name

COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size, isVideo = False):
    """Preprocess the input image to feed to the TFLite model"""
    if not isVideo:
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
    else:
        img = image_path

    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Retur the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    boxes = get_output_tensor(interpreter, 1)
    count = int(get_output_tensor(interpreter, 2))
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    
    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5, isVideo=False):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    if not isVideo:
        preprocessed_image, original_image = preprocess_image(image_path, (input_height, input_width))
    else:
        preprocessed_image, original_image = preprocess_image(image_path, (input_height, input_width), True)

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        
        label = f"{classes[class_id]} [{round(float(obj['score']) * 100)}%]"
        
        cv2.putText(original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8
    #return original_image_np


def inference(model_path: str, path_to_file: str, detection_threshold: str = .3):
    im = Image.open(path_to_file)
    im.thumbnail((512, 512), Image.ANTIALIAS)
    im.save(path_to_file, "PNG")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        path_to_file,
        interpreter,
        threshold=detection_threshold
    )

    # Show the detection result
    img = Image.fromarray(detection_result_image)
    img.show()
    img.save(path_to_file)

def inference_video(model_path: str, path_to_file: str, detection_threshold: str = .3):
    vidcap = cv2.VideoCapture(path_to_file)
    success,image = vidcap.read()
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    success = True
    count = 0
    while success:
        success,image = vidcap.read()
        detection_result_image = run_odt_and_draw_results(
        image,
        interpreter,
        threshold=detection_threshold, 
        isVideo = True
        )
        print(count)
        count += 1
        # Show the detection result
        #img = Image.fromarray(detection_result_image)
        
        #img.show()
        #opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow('object detection', detection_result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference_video(MODEL_PATH, "D:\\Projects\\Datasets\\drone\\video\\nexus\\video1\\video.mp4")
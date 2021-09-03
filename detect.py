
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))

CUSTOM_MODEL_NAME = 'my_ssd_mobilnet'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME)
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

print()
# %%
# Load pipeline config and build a detection model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Trying loading the pipeline.config...", end='\t')
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
print("[LOADED]")
print()

# %%
# Restore checkpoint
# ~~~~~~~~~~~~~~~~~~~~
print("Trying restoring the checkpoint...", end='\t')
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-15')).expect_partial()
print("[RESTORED]")
print()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Trying loading the label_map...", end='\t')
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'], use_display_name=True)
print("[LOADED]")
print()

# %%
# Define the video stream
# ~~~~~~~~~~~~~~~~~~~~~~~
print("Trying opening the camera...", end='\t')
cap = cv2.VideoCapture(14)
print("[OPENED]")
print()

frame_w = 640
frame_h = 480
set_res(cap, frame_w,frame_h)

while cap.isOpened():
    # Read frame from camera
    ret, image_np = cap.read()
    image_np = cv2.flip(image_np, 1)

    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    # ~~~~~~~~~~~~~~
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # ~~~~~~~~~~~~~~
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes']+label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.6,
          agnostic_mode=False)

    # %%
    # Display output
    # ~~~~~~~~~~~~~~~
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600))) #800 600

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

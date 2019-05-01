#!/usr/bin/env python
# coding: utf-8
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug import secure_filename
import StringIO as StrIO
import base64
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image

app = Flask(__name__, static_folder=os.path.dirname(os.path.realpath(__file__)))

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.9.* or later!')

# ## Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation

# Where model is located.
FROZEN_GRAPH = 'model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# # Detection
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


class Result:
    def __init__(self, category, score):
        self.category = category
        self.score = score

    def toJson(self):  
        return {           
            'category': self.category, 
            'score': self.score
        }

class ResultSummary:
    def __init__(self, results, image):
        self.results = results

        img_io = StrIO.StringIO()
        image.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)

        self.image = ("data:image/jpeg;base64," + str(base64.b64encode(img_io.read()).decode("utf-8")))
    
    def toJson(self):  
        return {           
            'results': [result.toJson() for result in self.results], 
            'image': self.image
        }

def generate_summary(image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4)
    im = Image.fromarray(image_np)

    # Generate results
    result_list = []
    for idx, score in enumerate(output_dict['detection_scores']):
        if score > 0:
            category_idx = output_dict['detection_classes'][idx]
            category = category_index.get(category_idx).get('name')
            result_list.append(Result(category, score * 100))
    summary = ResultSummary(result_list, im)
    return summary

@app.route('/')
def index():
   return app.send_static_file('index.html')

@app.route('/categories', methods=['GET'])
def categories():
    categories = [] 
    for idx in category_index:
        categories.append(category_index.get(idx).get('name'))
    categories.sort()
    return jsonify(categories)

@app.route('/classify', methods=['POST'])
def classify():
    img = Image.open(request.files['file'])
    
    # Maybe convert png->jpeg
    if not img.mode == 'RGB':
        img = img.convert('RGB')
    
    # Maybe downsize image
    max_size = 800, 800
    img.thumbnail(max_size, Image.ANTIALIAS)
    
    # Detect objects
    results = generate_summary(img)
    return jsonify({'summary':results.toJson()})

# @app.route('/imageobjectdetect', methods=['POST'])
# def image_object_detect2():
#     # Maybe downsize image
#     max_size = 600, 600
#     img_in = Image.open(request.files['file'])
#     img_in.thumbnail(max_size, Image.ANTIALIAS)
#     
#     # Detect objects
#     img_out = generate_summary(img_in)
#     img_io = StrIO.StringIO()
#     img_out.save(img_io, 'JPEG', quality=70)
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=5000, host=('0.0.0.0'))

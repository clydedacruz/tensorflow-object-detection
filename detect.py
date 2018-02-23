import base64
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
import sys

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

PATH_TO_CKPT = 'inference_graph\frozen_inference_graph.pb'
PATH_TO_LABELS = 'inference_graph\label_map.pbtxt'
MODEL_BASE = 'D:\Python\ML\models\research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '\object_detection')
sys.path.append(MODEL_BASE + '\slim')
from utils import label_map_util

PATH_TO_CKPT = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'


class ObjectDetector(object):

  def __init__(self):
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def draw_bounding_box_on_image(self,image, box, color='red', thickness=4):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=thickness, fill=color)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections

def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')
    boxes, scores, classes, num_detections = detector.detect(image)
    image.thumbnail((480, 480), Image.ANTIALIAS)
    # # show a bounding box 
    # detector.draw_bounding_box_on_image(image,boxes[0],thickness=int(scores[0]*10)-4)
    # image.show()


    # print("boxes:", boxes[0])
    # print("scores:", classes[0])
    # print("classes:", classes[0])
    # print("num_detections:", num_detections)

    print("\n ", image_path,"\t",scores[0])


detector = ObjectDetector()

TEST_IMAGE_DIR="test_images"
for i in range(6,10):
    detect_objects(TEST_IMAGE_DIR+"/"+"image"+str(i)+".jpg")


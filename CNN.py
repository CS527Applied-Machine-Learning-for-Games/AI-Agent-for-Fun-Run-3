import os
import tensorflow as tf
import numpy as np
import cv2
from utils import label_map_util
import time


class CNN(object):
    """
    This class acts as the intermediate "API" to the actual game. Double quotes API because we are not touching the
    game's actual code. It interacts with the game simply using screen-grab (input) and keypress simulation (output)
    using some clever python libraries.
    """
    # What model to download.
    MODEL_NAME = 'final_inference_graph'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('final_training', 'label_map.pbtxt')

    NUM_CLASSES = 6

    detection_graph = tf.Graph()

    def __init__(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)


    def nms_tf(bboxes,psocres,threshold):
        '''
        NMS: NMS using in-built tf.image.non_max_suppression(bboxes,scores,top_n_proposal_after_nms,iou_threshould)
            
        Input:
            bboxes(tensor of bounding proposals) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max)
            threshold(float): Overlapping threshold above which proposals will be discarded.
            
        Output:
            filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold.
        '''
        
        #First we need to convert bbox format from (x_min,y_min,x_max,y_max) to (y_min, x_min, y_max, x_max) ..
        #because tf.image.non_max_suppression method expects in that form.
        #For this we can use tf.unstack and tf.stack
        bboxes = tf.cast(bboxes,dtype=tf.float32)
        x_min,y_min,x_max,y_max = tf.unstack(bboxes,axis=1)
        bboxes = tf.stack([y_min,x_min,y_max,x_max],axis=1)
        bbox_indices = tf.image.non_max_suppression(bboxes,psocres, 100, iou_threshold=threshold)
        filtered_bboxes = tf.gather(bboxes,bbox_indices)
        scores = tf.gather(psocres,bbox_indices)
        y_min,x_min,y_max,x_max = tf.unstack(filtered_bboxes,axis=1)
        filtered_bboxes = tf.stack([x_min,y_min,x_max,y_max],axis=1)
        
        
        return filtered_bboxes

    def get_image_feature_map(self, image):
        start = time.time()
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                feature_vector = self.detection_graph.get_tensor_by_name("SecondStageBoxPredictor/Reshape_1:0")
                image_np = cv2.resize(np.array(image), (128, 128))
                image_np_expanded = np.expand_dims(image_np, axis=0)
                rep = sess.run([feature_vector], feed_dict={image_tensor: image_np_expanded})
                return np.array(rep).reshape(1, -1)

    

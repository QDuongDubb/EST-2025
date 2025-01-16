import os
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import math
import matplotlib.pyplot as plt
import cv2 as cv
import imageio
from scipy import misc

def load_saved_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            name, embedding_str = line.strip().split(': ')
            embedding = np.array(eval(embedding_str))
            embeddings_dict[name] = embedding
    return embeddings_dict


def align_image(image_path, image_size=160, margin=44, gpu_memory_fraction=1.0, detect_multiple_faces=False):
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  
    threshold = [0.6, 0.7, 0.7] 
    factor = 0.709  

    try:
        img = imageio.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        return None
    
    if img.ndim < 2:
        return None
    
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    
    img = img[:, :, 0:3]

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        img_size_np = np.asarray(img.shape)[0:2]
        det_arr = []

        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size_np / 2
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        
        aligned_image = []
        for det in det_arr:
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)

            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size_np[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size_np[0])

            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv.resize(cropped, (image_size, image_size))  
            scaled = scaled.astype(np.float32)
            scaled = (scaled - 127.5)
            scaled = np.expand_dims(scaled, axis=0)

            aligned_image.append(scaled)
            
        return aligned_image
    else:
        return None


def get_embedding(aligned_image, model_dir):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_dir, input_map=None)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
        
            preprocessed_image = np.squeeze(aligned_image)  
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        
            feed_dict = {images_placeholder: preprocessed_image, phase_train_placeholder: False}
            
            embedding = sess.run(embeddings, feed_dict=feed_dict)
            
            return embedding  
        
def calculate_distance(embedding_1, embedding_2, distance_metric=0):
    if distance_metric == 0:
        diff = np.subtract(embedding_1, embedding_2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        dot = np.sum(np.multiply(embedding_1, embedding_2), axis=1)
        norm = np.linalg.norm(embedding_1, axis=1) * np.linalg.norm(embedding_2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else: 
        raise ValueError('Undefined distance metric {}'.format(distance_metric))
    return dist


def find_closest_match(new_embedding, saved_embeddings):
    
    closest_name = None
    min_distance = float('inf')
    
    
    for name, saved_embedding in saved_embeddings.items():
        distance = calculate_distance(new_embedding, saved_embedding)
        print(name)
        print(distance)
        
        if distance < min_distance:
            min_distance = distance
            closest_name = name
            
    print(closest_name)
    
    return closest_name

def identify_face(new_embedding, embeddings_file_path):
    
    saved_embeddings = load_saved_embeddings(embeddings_file_path)   
    match = find_closest_match(new_embedding, saved_embeddings)
    return match

if __name__ == '__main__':
    image_path = 'res/test/image.png' 
    model_dir = 'models/20180402-114759'  
    embeddings_file_path = 'output_embedding/embeddings_1.txt'  
    aligned_image = align_image(image_path)
    
    if aligned_image is not None:
        input_embedding = get_embedding(aligned_image, model_dir) 
        
        identified_name = identify_face(input_embedding, embeddings_file_path)
        
        if identified_name:
            print('Face identified as:', identified_name)
        else:
            print('No matching face found.')
    else:
        print('No aligned images to process.')


import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import facenet


def load_image(image_path, image_size=160):
    img = cv.imread(image_path)  
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    img = cv.resize(img, (image_size, image_size))  
    img = img.astype(np.float32)  
    img = (img - 127.5)  
    img = np.expand_dims(img, axis=0)  

    return img  


def get_embedding(image_path):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            model_dir = "models/20180402-114759"  
            facenet.load_model(model_dir, input_map=None)
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
        
            preprocessed_image = load_image(image_path)
            print(preprocessed_image)
        
            feed_dict = {images_placeholder: preprocessed_image, phase_train_placeholder: False}
            
            embedding = sess.run(embeddings, feed_dict=feed_dict)
            
            return embedding  


image_dir = "res/test"

image_paths = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))

embeddings_dict = {}

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    embedding = get_embedding(image_path)
    embeddings_dict[image_name] = embedding

output_file = "output_embedding/embeddings_2.txt"
with open(output_file, "w") as f:
    for image_name, embedding in embeddings_dict.items():
        f.write("{}: {}\n".format(image_name, embedding.tolist()))

output_file

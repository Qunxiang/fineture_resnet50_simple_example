# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:58:54 2018

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os
import tensorflow as tf

from tensorflow.contrib.slim import nets

import generate_train_data

slim = tf.contrib.slim

flags = tf.flags

flags.DEFINE_string('images_dir', None, 'Path to training images directory.')
flags.DEFINE_string('output_dir', None, 'Path to directory to save model.')
flags.DEFINE_string('checkpoint_path', None, 'Path to the pretrained model.')
flags.DEFINE_integer('num_images', 10000, 'Number of images to be generated.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('num_steps', 500, 'Number of training steps.')

FLAGS = flags.FLAGS


def get_next_batch(batch_size=64, images_dir='./images'):
       """Get a batch set of training data.
       
       Args:
           batch_size: An integer representing the batch size.
           ...: Additional arguments.

       Returns:
           images: A 4-D numpy array with shape [batch_size, height, width, 
               num_channels] representing a batch of images.
           labels: A 1-D numpy array with shape [batch_size] representing
               the groundtruth labels of the corresponding images.
               
       Raises:
           ValueError: If images_dir is not exist.
       """
       if not os.path.exists(images_dir):
           raise ValueError('`images_dir` is not exist.')
       
       images = []
       labels = []
       image_files = np.array(glob.glob(os.path.join(images_dir, '*.jpg')))
       batch_size = min(batch_size, len(image_files))
       selected_indices = np.random.choice(len(image_files), batch_size)
       selected_images_path = image_files[selected_indices]
       for image_path in selected_images_path:
           image = cv2.imread(image_path)
           image = cv2.resize(image, (224, 224))
           label = image_path.split('_')[-1].split('.')[0]
           images.append(image)
           labels.append(int(label))
       images = np.array(images)
       labels = np.array(labels)
       return images, labels


def main(_):
    # Specify which gpu to be used
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    num_steps = FLAGS.num_steps
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is None:
        raise ValueError('`checkpoint_path` must be specified.')
    model_save_dir = FLAGS.output_dir
    if model_save_dir is None:
        model_save_dir = './training'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_path = os.path.join(model_save_dir, 'model.ckpt')
        
    # Generate training data
    images_dir = FLAGS.images_dir
    if images_dir is None:
        images_dir = './images'
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
        generate_train_data.generate_images(FLAGS.num_images, images_dir)
        print('Generate training images successfully.')

    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, num_classes=None,
                                                     is_training=is_training)
        
    with tf.variable_scope('Logits'):
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        logits = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='fc')
        
    checkpoint_exclude_scopes = 'Logits'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [
            scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)
            
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)
    
    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    
    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())
    
#    config = tf.ConfigProto(allow_soft_placement = True) 
#    config.gpu_options.per_process_gpu_memory_fraction = 0.95
#    with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(init)
        
        # Load the pretrained checkpoint file xxx.ckpt
        saver_restore.restore(sess, checkpoint_path)
        
        for i in range(num_steps):
            images, groundtruth_lists = get_next_batch(batch_size, images_dir)
            train_dict = {inputs: images, 
                          labels: groundtruth_lists,
                          is_training: True}

            sess.run(train_step, feed_dict=train_dict)
            
            loss_, acc_ = sess.run([loss, accuracy], feed_dict=train_dict)
            
            train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                i+1, loss_, acc_)
            print(train_text)
                
            if i == num_steps -1 or (i+1) % 1000 == 0:
                saver.save(sess, model_save_path, global_step=i+1)
                print('save mode to {}'.format(model_save_dir))
                

if __name__ == '__main__':
    tf.app.run()
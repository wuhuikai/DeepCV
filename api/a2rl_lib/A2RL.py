from __future__ import absolute_import

import pickle
import numpy as np
import tensorflow as tf
import skimage.io as io

import network

from actions import command2action, generate_bbox, crop_input


global_dtype = tf.float32
with open('models/auto_crop/vfn_rl.pkl', 'rb') as f:
    var_dict = pickle.load(f)

image_placeholder = tf.placeholder(dtype=global_dtype, shape=[None,227,227,3])
global_feature_placeholder = network.vfn_rl(image_placeholder, var_dict)

h_placeholder = tf.placeholder(dtype=global_dtype, shape=[None,1024])
c_placeholder = tf.placeholder(dtype=global_dtype, shape=[None,1024])
action, h, c = network.vfn_rl(image_placeholder, var_dict, global_feature=global_feature_placeholder,
                                                           h=h_placeholder, c=c_placeholder)
sess = tf.Session()


def auto_cropping(path):
    origin_image = [io.imread(path).astype(np.float32) / 255 - 0.5]

    terminals = np.zeros(1)
    ratios = np.asarray([[0, 0, 20, 20]])
    img = crop_input(origin_image, generate_bbox(origin_image, ratios))

    global_feature = sess.run(global_feature_placeholder, feed_dict={image_placeholder: img})
    h_np = np.zeros([1, 1024])
    c_np = np.zeros([1, 1024])

    for _ in range(15):
        action_np, h_np, c_np = sess.run((action, h, c), feed_dict={image_placeholder: img,
                                                                    global_feature_placeholder: global_feature,
                                                                    h_placeholder: h_np,
                                                                    c_placeholder: c_np})
        ratios, terminals = command2action(action_np, ratios, terminals)
        bbox = generate_bbox(origin_image, ratios)
        if np.sum(terminals) == 1:
            break

        img = crop_input(origin_image, bbox)

    return bbox[0]

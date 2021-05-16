import os
import glob
import csv
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def process_image(feed, width, height):
    """
    :param feed: image/s location and corresponding label.
    :param width: width of image.
    :param height: height of image.
    :return: This function returns the processed and resized image data.
    """
    label = feed[1]
    image_string = tf.read_file(feed[0])
    image_decoded = tf.io.decode_image(image_string, 3, expand_animations=False)
    image_decoded = tf.image.random_crop(image_decoded, [height, width, 3])
    image_decoded = tf.image.random_hue(image_decoded, 0.02)
    image_decoded = tf.image.random_brightness(image_decoded, 0.7)
    image_decoded = tf.image.random_contrast(image_decoded, 0.0, 0.7)
    image_decoded = tf.image.random_saturation(image_decoded, 0.0, 0.3)
    image_decoded.set_shape((height, width, 3))
    return image_decoded, label


def build_inputs(specs, classes, shuffle=True, num_threads=4):
    """
    :param specs: dictionary of info on input and output data.
    :param classes: number of label classes in dataset.
    :param shuffle: randomize batches.
    :param num_threads: tf.train.batch threads to use.
    :return: This function returns a tensor batch queue from files provided.
    """
    image_feed = np.array(glob.glob(os.path.join(specs['dir'], '**', '*.jpg*'), recursive=True))
    images, labels = [], []
    with open(os.path.join(specs['dir'], "labels_training.csv"), 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[1] in specs['labels']:
                if row[0] in image_feed:
                    images.append(row[0])
                    one_hot = []
                    for i in range(classes):
                        one_hot.append(0 if i != specs['labels'][row[1]] else 1)
                    labels.append(one_hot)
    images_list = tf.constant(images)
    labels_list = tf.constant(labels)
    input_queue = tf.train.slice_input_producer([images_list, labels_list], shuffle=shuffle)
    image, label = process_image(input_queue, specs['x_size'], specs['y_size'])
    image_batch, label_batch = tf.train.batch([image, label], batch_size=specs['batch_size'], num_threads=num_threads, capacity=specs['batch_size']*2)
    return image_batch, label_batch


def build_log_dir(main_name, path_name=None, second_path_name=None):
    """
    :param main_name: top folder name.
    :param path_name: child folder name.
    :param second_path_name: child within a child folder name.
    :return: Returns path to main folder and creates the folder.
    """
    log_path = main_name
    if second_path_name is not None:
        log_path = os.path.join(log_path, path_name)
        log_path = os.path.join(log_path, second_path_name)
    elif path_name is not None:
        log_path = os.path.join(log_path, path_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path


def tf_config_setup():
    """
    :return: Returns tensorflow configuration.
    """
    tf_config = tf.ConfigProto(allow_soft_placement=False)
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.polling_inactive_delay_msecs = 50
    return tf_config


def initialization(sess):
    """
    :param sess: tensorflow active session.
    :return: Returns the initialized tensorflow session.
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    return coord, threads

from .util import *
from tqdm import tqdm


class DenseNet121:
    def __init__(self, specs, location, classes, pretrained=False):
        """
        :param specs: dictionary of info on input and output data.
        :param location: directory path to save to.
        :param classes: number of classes to classify.
        :param pretrained: boolean to load a pretrained densenet121.
        """
        self.specs = specs
        self.location = location
        self.classes = classes
        self.pretrained = pretrained
        self.nesterov = True

    def graph(self, target, training):
        """
        :param target: embedded data to reconstruct.
        :param training: batch normalization boolean.
        :return: This function returns the DenseNet121 graph.
        """
        def conv_layer(input_layer, filter_size, kernel_size, name):
            with tf.variable_scope(name):
                output_layer = tf.layers.batch_normalization(input_layer, training=training)
                output_layer = tf.nn.relu(output_layer)
                return tf.layers.conv2d(output_layer, filters=filter_size, kernel_size=kernel_size, strides=1, padding="SAME")

        def block_layer(input_layer, block_size, name):
            with tf.variable_scope('dense_block%d' % name):
                for i in range(block_size):
                    previous_layer = input_layer if i == 0 else encoding
                    num_filters = 64
                    with tf.variable_scope('conv_block%d' % (i + 1)):
                        encoding = conv_layer(previous_layer, num_filters * 4, 1, 'x1')
                        encoding = conv_layer(encoding, num_filters, 3, 'x2')
                        encoding = tf.concat([encoding, previous_layer], axis=3)
                    num_filters += 32
            if name != 4:
                with tf.variable_scope('transition_block%d' % name):
                    encoding = conv_layer(encoding, 32, 1, 'blk')
                    encoding = tf.nn.avg_pool(encoding, 2, 2, "VALID")
            return encoding

        with tf.variable_scope('densenet121', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('layers'):
                target = tf.layers.conv2d_transpose(target, filters=64, kernel_size=7, strides=2, padding="SAME", name='conv1')
                target = tf.layers.batch_normalization(target, training=training)
                target = tf.nn.relu(target)
                target = tf.nn.max_pool(target, ksize=3, strides=2, padding='SAME')
                target = block_layer(target, 6, 1)
                target = block_layer(target, 12, 2)
                target = block_layer(target, 24, 3)
                target = block_layer(target, 16, 4)
                target = tf.layers.batch_normalization(target, training=training)
                target = tf.nn.relu(target)
                target = tf.nn.avg_pool(target, 7, 1, "VALID")
                target = tf.reduce_mean(target, axis=[1, 2], keep_dims=True)
            with tf.variable_scope('classifier'):
                logits = tf.layers.dense(target, self.classes)
                logits = tf.identity(logits, 'logits')
                return logits

    def optimize(self, loss, scope, learning_rate):
        """
        :param loss: the classifier loss.
        :param scope: the model's variable scope.
        :param learning_rate: learning rate of the optimizer.
        :return: This function returns a model's optimizer.
        """
        ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
        with tf.control_dependencies(ops):
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=self.nesterov).minimize(loss, var_list=var)

    def ops(self, target_image, target_label, training, learning_rate):
        """
        :param target_image: ground truth image.
        :param target_label: ground truth label.
        :param training: batch normalization boolean.
        :param learning_rate: learning rate of the optimizer.
        :return: This function returns the generator graph of the model and the reconstructed and encoded outputs.
        """
        logit = self.graph(target_image, training)
        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=target_label))
        graph_ops = self.optimize(classification_loss, 'densenet121', learning_rate)
        return graph_ops

    def freeze(self):
        tf.reset_default_graph()
        get_image_batch, get_label_batch = build_inputs(self.specs, self.classes)
        target_image = tf.placeholder(tf.float32, [None, None, None, 3], name='target_image_ph')
        logit = self.graph(target_image, False)
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            atr_variables = [var for var in tf.global_variables() if var.name.startswith('densenet121/layers') or var.name.startswith('densenet121/classifier')]
            atr_saver = tf.train.Saver(atr_variables)
            atr_saver.restore(sess, os.path.join(self.location, 'atr.ckpt'))
            for _ in range(1):
                target_image_batch, target_label_batch = sess.run([get_image_batch, get_label_batch])
                graph_feed = {target_image: target_image_batch}
                logit_output = sess.run(logit, graph_feed)
                accuracy = 0
                for out in range(self.specs['batch_size']):
                    logit_index = np.argmax(logit_output[out])
                    true_index = np.argmax(target_label_batch[out])
                    accuracy += 1 if logit_index == true_index else 0
            print("ATR accuracy: " + str(accuracy / (self.specs['batch_size']) * 100) + "%")
            logits_node_name = ['densenet121/classifier/logits']
            atr_frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, logits_node_name)
            with open(os.path.join(self.location, 'atr_graph.pb'), 'wb') as f:
                f.write(atr_frozen_graph_def.SerializeToString())
            coord.request_stop()
            coord.join(threads)
        return

    def train(self):
        """
        :return: This function runs the model training.
        """
        if self.pretrained:
            tf.reset_default_graph()
        get_image_batch, get_label_batch = build_inputs(self.specs, self.classes)
        target_image = tf.placeholder(tf.float32, [None, None, None, 3], name='target_image_ph')
        target_label = tf.placeholder(tf.float32, [None, self.classes], name='target_label_ph')
        learning_rate = tf.placeholder(tf.float32, shape=[])
        training = tf.placeholder(tf.bool, name='training')
        graph_ops = self.ops(target_image, target_label, training, learning_rate)
        with tf.Session(config=tf_config_setup()) as sess:
            coord, threads = initialization(sess)
            dn121_variables = [var for var in tf.global_variables() if var.name.startswith('densenet121/layers')]
            dn121_saver = tf.train.Saver(dn121_variables)
            lr = self.specs['lr']
            if self.pretrained:
                dn121_saver.restore(sess, os.path.join(self.location, 'dn121.ckpt'))
                atr_saver = tf.train.Saver()
            for i in tqdm(range(0, self.specs['epochs'])):
                if self.pretrained and i != 0:
                    lr *= 0.05
                if not self.pretrained and i != 0:
                    # dn121_saver.save(sess, os.path.join(self.location, 'dn121.ckpt'))
                    lr **= 0.97
                    # print("Model saved at epoch %d" % i)
                target_image_batch, target_label_batch = sess.run([get_image_batch, get_label_batch])
                graph_feed = {target_image: target_image_batch, target_label: target_label_batch, training: True, learning_rate: lr}
                _ = sess.run(graph_ops, graph_feed)
            if self.pretrained:
                atr_saver.save(sess, os.path.join(self.location, 'atr.ckpt'))
            else:
                dn121_saver.save(sess, os.path.join(self.location, 'dn121.ckpt'))
            coord.request_stop()
            coord.join(threads)
        return

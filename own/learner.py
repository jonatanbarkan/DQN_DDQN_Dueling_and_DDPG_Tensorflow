from __future__ import division, absolute_import, print_function
import numpy as np
import tensorflow as tf
from model.general_model.build_object_models import Build


class Learner(Build):

    def __init__(self, x, params, sess=None, built=False, experience=False):
        self.name = 'Learner'
        super(Learner, self).__init__(x, params)
        if sess is not None:
            sess.as_default()
            sess.graph.as_default()
        if not built:  # new learner
            self.local_step = tf.Variable(0, name='local_step', trainable=False)
            self.experience_step = None
            if experience:
                self.experience_step = tf.Variable(0, name='experience', trainable=False)
            self.in_training = tf.placeholder(tf.bool, name="phase")
            with tf.name_scope('label_layer'):
                self.label = tf.placeholder_with_default(input=tf.zeros([1] + self.input_shape[1:]), shape=self.input_shape, name='label')
                # self.direction = tf.placeholder_with_default(['stay'], shape=[1], name='direction')
                # self.label = self.tf_image_shift(self.not_shifted_label, self.direction, self.kernels_shapes[0][0])
            with tf.name_scope('Inference'):
                if "with_batch_norm" in params:
                    batch_norm_dict = params["with_batch_norm"]
                else:
                    batch_norm_dict = None
                if "normalize_input" in params:
                    normalize_input = params["normalize_input"]
                else:
                    normalize_input = False
                if "noisy_input" in params:
                    add_noise = params["noisy_input"]
                else:
                    add_noise = None
                self.output, self.network_tensors = self.inference(batch_norm_dict, normalize_input, add_noise)
            with tf.name_scope('loss'):
                self.cost, self.epsilon_squared = self.loss(predictions=self.output, labels=self.label)
            with tf.name_scope('train'):
                self.train_op, self.learning_rate, self.momentum, self.experience_train_op = self.train(self.cost)
            # with tf.name_scope('epsilon_gradients') as scope:
            # self.epsilon_gradients = tf.gradients(self.cost, [self.epsilon_squared])[0]

        else:  # load learner
            #  to get tensor names run sess.graph.get_operations()
            # self.get_ops_by_scopes(sess, 'Agent', 'Learner', 'loss')
            self.label = sess.graph.get_tensor_by_name('Agent/Learner/label_layer/label:0')
            self.output = sess.graph.get_tensor_by_name('Agent/Learner/Inference/output_layer/convolution/layer_activation:0')
            self.cost = sess.graph.get_tensor_by_name('Agent/Learner/loss/Mean:0')
            self.epsilon_squared = sess.graph.get_tensor_by_name('Agent/Learner/loss/epsilon_squared_online:0')
            # self.epsilon_gradients = sess.graph.get_tensor_by_name('Agent/Learner/epsilon_gradients/grads:0')
            self.train_op = sess.graph.get_tensor_by_name('Agent/Learner/train/train_op:0')
            self.learning_rate = sess.graph.get_tensor_by_name('Agent/Learner/train/learning_rate:0')
            self.momentum = sess.graph.get_tensor_by_name('Agent/Learner/train/Momentum:0')
            self.experience_train_op = None
            self.experience_step = None
            if experience:
                self.experience_train_op = sess.graph.get_tensor_by_name('Agent/Learner/train/experience_train_op:0')
                self.experience_step = sess.graph.get_tensor_by_name('Agent/Learner/experience:0')
            self.local_step = sess.graph.get_tensor_by_name('Agent/Learner/local_step:0')
            self.in_training = sess.graph.get_tensor_by_name('Agent/Learner/phase:0')
            from collections import OrderedDict as Odict
            self.network_tensors = Odict()
            for hid in range(self.num_of_hidden_layers) + [-1]:
                self.network_tensors[hid] = dict()
                if hid == 0:
                    layer_name = 'input_layer'
                elif hid == -1:
                    layer_name = 'output_layer'
                else:
                    layer_name = 'hidden_layer_{0}'.format(hid + 1)
                self.network_tensors[hid]['w'] = sess.graph.get_tensor_by_name('Agent/Learner/Inference/{}/weights/Variable:0'.format(layer_name))
                self.network_tensors[hid]['b'] = sess.graph.get_tensor_by_name('Agent/Learner/Inference/{}/bias/Variable:0'.format(layer_name))

    # def shift(self, direction, step_size):
    #     self.label = self.tf_image_shift(self.label, direction, step_size)

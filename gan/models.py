# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)

"""Contains the base class for models."""
class BaseModel(object):
	"""Inherit from this class when implementing new models."""

	def create_model(self, **unused_params):
		"""Define variables of the model."""
		raise NotImplementedError()

	def run_model(self, unused_model_input, **unused_params):
		"""Run model with given input."""
		raise NotImplementedError()

	def get_variables(self):
		"""Return all variables used by the model for training."""
		raise NotImplementedError()

class SampleGenerator(BaseModel):
	def __init__(self):
		self.noise_input_size = 100

	def create_model(self, output_size, **unused_params):
		h1_size = 512
		h2_size = 512
		h3_size = output_size
		h4_size = output_size

		self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
		self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')

		self.G_W2 = tf.Variable(xavier_init([h1_size, h2_size]), name='g/w2')
		self.G_b2 = tf.Variable(tf.zeros(shape=[h2_size]), name='g/b2')

		self.G_W3 = tf.Variable(xavier_init([h2_size, h3_size]), name='g/w3')
		self.G_b3 = tf.Variable(tf.zeros(shape=[h3_size]), name='g/b3')

		# self.G_W4 = tf.Variable(xavier_init([h3_size, output_size]), name='g/w4')
		# self.G_b4 = tf.Variable(tf.zeros(shape=[output_size]), name='g/b4')

	def run_model(self, model_input, is_training=True, **unused_params):
		net = model_input
		
		net = tf.contrib.layers.batch_norm(net, is_training=is_training)
		net = tf.nn.sigmoid(tf.matmul(net, self.G_W1) + self.G_b1)

		net = tf.contrib.layers.batch_norm(net, is_training=is_training)
		net = tf.nn.sigmoid(tf.matmul(net, self.G_W2) + self.G_b2)

		net = tf.contrib.layers.batch_norm(net, is_training=is_training)
		net = tf.nn.sigmoid(tf.matmul(net, self.G_W3) + self.G_b3)

		# net = tf.contrib.layers.batch_norm(net, is_training=is_training)
		# net = tf.nn.tanh(tf.matmul(net, self.G_W4) + self.G_b4)
		
		return {"output": net}

	def get_variables(self):
		return [self.G_W1, self.G_b1,
				self.G_W2, self.G_b2,
				self.G_W3, self.G_b3]

class SampleDiscriminator(BaseModel):
	# def create_model(self, input_size, **unused_params):
	# 	h1_size = 128
	# 	self.D_W1 = tf.Variable(xavier_init([input_size, h1_size]), name='d/w1')
	# 	self.D_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='d/b1')

	# 	self.D_W2 = tf.Variable(xavier_init([h1_size, 1]), name='d/w2')
	# 	self.D_b2 = tf.Variable(tf.zeros(shape=[1]), name='d/b2')

	# def run_model(self, model_input, is_training=True, **unused_params):
	# 	net = tf.nn.relu(tf.matmul(model_input, self.D_W1) + self.D_b1)
	# 	logits = tf.matmul(net, self.D_W2) + self.D_b2
	# 	predictions = tf.nn.sigmoid(logits)
	# 	return {"logits": logits, "predictions": predictions}

	# def get_variables(self):
	# 	return [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

	def create_model(self, input_size, **unused_params):
		h1_size = 10816
		
		self.D_W_conv11 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='D_W_conv11')
		# self.D_b_conv11 = tf.Variable(tf.random_normal([32]), name='D_b_conv11')
		self.D_W_conv12 = tf.Variable(tf.random_normal([3, 3, 32, 32]), name='D_W_conv12')
		# self.D_b_conv12 = tf.Variable(tf.random_normal([32]), name='D_b_conv12')

		self.D_W_conv21 = tf.Variable(tf.random_normal([3, 3, 32, 64]), name='D_W_conv21')
		# self.D_b_conv21 = tf.Variable(tf.random_normal([64]), name='D_b_conv21')
		self.D_W_conv22 = tf.Variable(tf.random_normal([3, 3, 64, 64]), name='D_W_conv22')
		# self.D_b_conv22 = tf.Variable(tf.random_normal([64]), name='D_b_conv22')

		self.D_W_conv31 = tf.Variable(tf.random_normal([3, 3, 64, 128]), name='D_W_conv31')
		self.D_W_conv32 = tf.Variable(tf.random_normal([3, 3, 128, 128]), name='D_W_conv32')

		self.D_W_conv41 = tf.Variable(tf.random_normal([3, 3, 128, 256]), name='D_W_conv41')
		self.D_W_conv42 = tf.Variable(tf.random_normal([3, 3, 256, 256]), name='D_W_conv42')

		self.D_W_conv51 = tf.Variable(tf.random_normal([3, 3, 256, 256]), name='D_W_conv51')
		self.D_W_conv52 = tf.Variable(tf.random_normal([3, 3, 256, 256]), name='D_W_conv52')

		self.D_W_fc1 = tf.Variable(tf.random_normal([1024, 1]), name='D_W_fc1')
		self.D_b_fc1 = tf.Variable(tf.random_normal([1]), name='D_b_fc1')
		# self.D_W_fc2 = tf.Variable(tf.random_normal([256, 1]), name='D_W_fc2')
		# self.D_b_fc2 = tf.Variable(tf.random_normal([1]), name='D_b_fc2')

	def run_model(self, model_input, is_training=True, **unused_params):
		strides = 1

		print(model_input.shape)
		net = tf.reshape(model_input, shape=[-1, 50, 50, 1])

		net = tf.nn.conv2d(net, self.D_W_conv11, strides=[1, strides, strides, 1], padding='SAME')
		# net = tf.nn.bias_add(net, self.D_b_conv11)
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv12, strides=[1, strides, strides, 1], padding='SAME')
		# net = tf.nn.bias_add(net, self.D_b_conv12)
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(net.shape)

		net = tf.nn.conv2d(net, self.D_W_conv21, strides=[1, strides, strides, 1], padding='SAME')
		# net = tf.nn.bias_add(net, self.D_b_conv21)
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv22, strides=[1, strides, strides, 1], padding='SAME')
		# net = tf.nn.bias_add(net, self.D_b_conv22)
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(net.get_shape)

		net = tf.nn.conv2d(net, self.D_W_conv31, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv32, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(net.get_shape)

		net = tf.nn.conv2d(net, self.D_W_conv41, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv42, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(net.get_shape)

		net = tf.nn.conv2d(net, self.D_W_conv51, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv52, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print(net.get_shape)


		
		net = tf.reshape(net, [-1, 1024])
		print(net.shape)

		net = tf.add(tf.matmul(net, self.D_W_fc1), self.D_b_fc1)
		net = tf.nn.sigmoid(net)

		# net = tf.add(tf.matmul(net, self.D_W_fc2), self.D_b_fc2)
		# net = tf.nn.sigmoid(net)

		logits = net
		predictions = tf.nn.sigmoid(logits)
		print("_______________________________________________________________")
		return {"logits": logits, "predictions": predictions}

	def get_variables(self):
		# return [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
		return [self.D_W_conv11,
				self.D_W_conv12,
				self.D_W_conv21,
				self.D_W_conv22,
				self.D_W_fc1,
				self.D_b_fc1]
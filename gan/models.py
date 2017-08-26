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
		self.G_W_deconv11 = tf.Variable(tf.random_normal([1, 1, 100, 1]), name='G_W_deconv11')
		self.G_b_deconv11 = tf.Variable(tf.random_normal([100]), name='G_b_deconv11')
		self.G_W_deconv12 = tf.Variable(tf.random_normal([5, 5, 1, 100]), name='G_W_deconv12')
		self.G_b_deconv12 = tf.Variable(tf.random_normal([1]), name='G_b_deconv12')

		# self.G_W_fc1 = tf.Variable(xavier_init([h1_size, output_size]), name='g/w2')
		# self.G_b_fc1 = tf.Variable(tf.zeros(shape=[output_size]), name='g/b2')


	def run_model(self, model_input, is_training=True, **unused_params):
		print("_______________________________________________________________")
		strides = 1
		
		print(model_input.shape)
		net = tf.reshape(model_input, shape=[-1, 10, 10, 1])
		print(net.shape)
		net = tf.nn.conv2d_transpose(net, self.G_W_deconv11, [-1, 10, 10, 100], strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.bias_add(net, self.G_b_deconv11)
		net = tf.nn.tanh(net)
		print(net.shape)

		net = tf.nn.conv2d_transpose(net, self.G_W_deconv12, [-1, 50, 50, 1], strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.bias_add(net, self.G_b_deconv12)
		net = tf.nn.tanh(net)
		output = net
		print(net.shape)

		# net = tf.nn.relu(tf.matmul(model_input, self.G_W1) + self.G_b1)
		# output = tf.nn.sigmoid(tf.matmul(net, self.G_W2) + self.G_b2)
		
		net = tf.reshape(net, [-1])

		# net = tf.add(tf.matmul(net, self.G_W_fc1), self.G_b_fc1)
		# net = tf.nn.sigmoid(net)

		output = net
		print(output.shape)
		print("_______________________________________________________________")
		return {"output": output}

	def get_variables(self):
		return [self.G_W_deconv11,
				self.G_b_deconv11,
				self.G_W_deconv12,
				self.G_b_deconv12]

	############################################################################
	# def create_model(self, output_size, **unused_params):
	# 	h1_size = 128
	# 	self.G_W1 = tf.Variable(xavier_init([self.noise_input_size, h1_size]), name='g/w1')
	# 	self.G_b1 = tf.Variable(tf.zeros(shape=[h1_size]), name='g/b1')

	# 	self.G_W2 = tf.Variable(xavier_init([h1_size, output_size]), name='g/w2')
	# 	self.G_b2 = tf.Variable(tf.zeros(shape=[output_size]), name='g/b2')

	# def run_model(self, model_input, is_training=True, **unused_params):
	# 	print("_______________________________________________________________")		
	# 	print(model_input.shape)

	# 	net = tf.nn.sigmoid(tf.matmul(model_input, self.G_W1) + self.G_b1)
	# 	print(net.shape)
	# 	output = tf.nn.sigmoid(tf.matmul(net, self.G_W2) + self.G_b2)

	# 	print(output.shape)
	# 	print("_______________________________________________________________")
	# 	return {"output": output}

	# def get_variables(self):
	# 	return [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

class SampleDiscriminator(BaseModel):
	def create_model(self, input_size, **unused_params):
		h1_size = 10816
		
		self.D_W_conv11 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='D_W_conv11')
		self.D_b_conv11 = tf.Variable(tf.random_normal([32]), name='D_b_conv11')
		self.D_W_conv12 = tf.Variable(tf.random_normal([3, 3, 32, 32]), name='D_W_conv12')
		self.D_b_conv12 = tf.Variable(tf.random_normal([32]), name='D_b_conv12')

		self.D_W_conv21 = tf.Variable(tf.random_normal([3, 3, 32, 64]), name='D_W_conv21')
		self.D_b_conv21 = tf.Variable(tf.random_normal([64]), name='D_b_conv21')
		self.D_W_conv22 = tf.Variable(tf.random_normal([3, 3, 64, 64]), name='D_W_conv22')
		self.D_b_conv22 = tf.Variable(tf.random_normal([64]), name='D_b_conv22')

		self.D_W_fc1 = tf.Variable(tf.random_normal([10816, 256]), name='D_W_fc1')
		self.D_b_fc1 = tf.Variable(tf.random_normal([256]), name='D_b_fc1')
		self.D_W_fc2 = tf.Variable(tf.random_normal([256, 1]), name='D_W_fc2')
		self.D_b_fc2 = tf.Variable(tf.random_normal([1]), name='D_b_fc2')

	def run_model(self, model_input, is_training=True, **unused_params):
		strides = 1

		print(model_input.shape)
		net = tf.reshape(model_input, shape=[-1, 50, 50, 1])
		net = tf.nn.conv2d(net, self.D_W_conv11, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.bias_add(net, self.D_b_conv11)
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv12, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.bias_add(net, self.D_b_conv12)
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		net = tf.nn.conv2d(net, self.D_W_conv21, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.bias_add(net, self.D_b_conv21)
		net = tf.nn.sigmoid(net)
		net = tf.nn.conv2d(net, self.D_W_conv22, strides=[1, strides, strides, 1], padding='SAME')
		net = tf.nn.bias_add(net, self.D_b_conv22)
		net = tf.nn.sigmoid(net)
		net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		
		net = tf.reshape(net, [-1, 10816])
		print(net.shape)

		net = tf.add(tf.matmul(net, self.D_W_fc1), self.D_b_fc1)
		net = tf.nn.sigmoid(net)

		net = tf.add(tf.matmul(net, self.D_W_fc2), self.D_b_fc2)
		net = tf.nn.sigmoid(net)

		logits = net
		predictions = tf.nn.sigmoid(logits)
		print("_______________________________________________________________")
		return {"logits": logits, "predictions": predictions}

	def get_variables(self):
		# return [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
		return [self.D_W_conv11,
				self.D_b_conv11,
				self.D_W_conv12,
				self.D_b_conv12,
				self.D_W_conv21,
				self.D_b_conv21,
				self.D_W_conv22,
				self.D_b_conv22,
				self.D_W_fc1,
				self.D_b_fc1,
				self.D_W_fc2,
				self.D_b_fc2]

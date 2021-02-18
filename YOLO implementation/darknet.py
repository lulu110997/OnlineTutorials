#!/usr/bin/env python3
# Based on the tutorial series by Ayoosh Kathuria 
# @ https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

class Darknet(nn.Module):
	'''
	Output of a YOLO algo is n x n x a x (5+C)
	n: number of grip
	a: number of anchors
	C: is the number of class
	5 + C because each cell outputs Pc, bx, by, bh, bw + # classes
	'''
	def __init__(self):
		super(Darknet, self).__init__()
		self.blocks = parse_cfg('cfg/yolov3.cfg')
		self.net_info, self.module_list = create_modules(self.blocks)

	def forward(self,x,CUDA):
		modules = self.blocks[1:]
		output_feature_maps = {} # Stores the feature maps of the previous layers in a dict
		write = 0
		for index, item in enumerate(modules):
			module_type = item['type']
			if module_type == 'convolutional' or module_type == 'upsample':
				x = self.module_list[index](x) # Feed the input in the network's layer

			elif module_type == 'route':
				layers = item['layers']
				layers = [int(a) for a in layers]

				# if layers[0] > 0:
				# 	layers[0] = layers[0] - index

				if len(layers) == 1: 
					# if only 1 number is in the list, obtain the feature map from the previous layers[0]
					x = output_feature_maps[index + layers[0]]
				else: 
					# Need to concatenate the feature maps from two previous layers 
					layers[1] = layers[1] - index

					map1 = output_feature_maps[i + layers[0]]
					map2 = output_feature_maps[i + layers[1]]
					# Concatenate the maps along the depth. Remember that the dim in pytorch is 
					# batch x depth x height x width
					x = torch.cat((map1,map2),1) 

			elif module_type == 'shortcut':
				# Skip connection		
				from_ = int(module['from'])
				x = outputs[index-1] + outputs[index+from_] 

class EmptyLayer(nn.Module):
	'''
	Dummy layer for the route and shortcut layers. Rather than defining the forward pass here, the 
	actual operation of the route and shortcut layer can be done in the forward pass of the darknet
	'''
	def __init__(self):
		super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
	'''
	Holds the anchors used to detect bounding boxes
	'''
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors


def parse_cfg(cfgfile):
	'''
	Takes a configuration file

	Returns a list of blocks. Each blocks describes a block in the neural 
	network to be built. Block is represented as a dictionary in the list

	Note how the first item of the dictionary, blocks, is going to be
	the network information

	The blocks is a list of dictionaries which has strings inside of them
	'''
	file = open(cfgfile, 'r')
	lines = file.read().split('\n') # This reads the whole file and stores each line in the file into a list
	lines = [x for x in lines if len(x) > 0] # Removes empty lines
	lines = [x for x in lines if x[0] != '#'] # Removes comments
	lines = [x.rstrip().lstrip() for x in lines] # Removes any trailing spaces

	block = {}
	blocks = []

	for line in lines:
		if line[0] == "[": # '[' Marks the start of a new block
			if len(block) != 0:
			# If block is not empty, implies it is storing values of previous block.
			# Save the content of block into the blocks list and re-init block
				blocks.append(block)
				block = {}
			# Adds a key in the dictionary called 'type' and gives it the value 
			# of the line from the start to the second last (avoids the ']').
			# Remove any trailing spaces in the line before saving it in the dict
			block["type"] = line[1:-1].rstrip() 
		else: # Handle if the line defines the value of a block's parameter
			# Obtain the parameter (eg activation func, padding, filter size) 
			# in the block and its value. Then save it in the blocks dict with
			# the parameter as the key and value as the value. Remove the 
			# parameter's trailing space and the value's leading space
			parameter,value = line.split("=")
			block[parameter.rstrip()] = value.lstrip()
	blocks.append(block)
	return blocks

def create_modules(blocks):
	'''
	Takes a list, blocks, returned by the parse_cfg function then we 
	extend the nn.Module class by creating code for YOLO, route and 
	shortcut. The code for convolutional and upsample is already
	provided by nn.Module

	Returns an nn.ModuleList, which is similar to a Python list that contains
	nn.Module objects but whatever is inside this list will become part of 
	the nn.Module module. Contains the operations for each layer of the network
	'''
	net_info = blocks[0] # Captures the info about the network (input and pre-processing)
	module_list = nn.ModuleList() # Store the block's parameters as a member of the nn.Module module
	prev_filters = 3 # Keeps track of the previous filter depth as the input for the next conv layer. Initialised
					 # as 3 because the initial input image is RGB and has depth 3
	output_filters = [] # Keeps track of the filter depth for each block in case the route laye brings in a 
						# concatenated feature map from previous layers
	for index, block in enumerate(blocks[1:]):
		# continue
	# Iterate through the each block of the network, not including the networks info
		module = nn.Sequential()
		# Check the type of block
		# Create a new module for the block
		# Append to module_list
		if block['type'] == 'convolutional':
			# Obtain layer information, turn them into an int if it's a number
			activation = block['activation'] # Activation function
			try: # If the conv layer uses batch normalization
				batch_normalize = int(block['batch_normalize']) # Batch normalisation input
				bias = False
			except: # If it doesn't
				batch_normalize = 0
				bias = True
			# Obtain the filter size, padding, kernel size and stride
			filters = int(block['filters'])
			padding = int(block['pad'])
			kernel_size = int(block['size'])
			stride = int(block['stride'])
			# Explanation on the paddings:
			 # https://github.com/pjreddie/darknet/issues/950
			 # https://github.com/pjreddie/darknet/issues/950
			 # Andrew Ng's coursera video on padding helped too
			if padding:
				pad = (kernel_size - 1) // 2 # When kernel size is odd and >1, o/p size is equal to i/p size
			else:
				pad = 0

			# Add convolutional operation	
			conv = nn.Conv2d(prev_filters, filters, kernel_size, pad, bias = bias)
			module.add_module("conv_{}".format(index), conv)

			# Add batch normalisation
			if batch_normalize:
				bn = nn.BatchNorm2d(filters)
				module.add_module("batch_norm_{}".format(index), bn)

			# Add activation layer
			if activation == "leaky":
				activation_fn = nn.LeakyReLU(0.1, inplace=True)
				module.add_module("leaky_{}".format(index), activation_fn)

		elif block['type'] == 'upsample': # for upsampling, use Bilinear2dUpsampling:
			stride = int(block["stride"])
			upsample = nn.Upsample(scale_factor = stride, mode="biliner")
			module.add_module("upsample{}".format(index), upsample)

		elif block['type'] == 'route': # Route layer
			route = EmptyLayer()
			module.add_module("route{}".format(index), route)

			# Need to split the string into a list with two elements,
			# then obtain the start and end values, converting the
			# strings into an integer type
			block['layers'] = block['layers'].split(',')
			start = int(block['layers'][0])
			try:
				end = int(block['layers'][1])
			except:
				end = 0

			# Positive annotation
			# if start > 0:
			# 	start = start - index
			# if end > 0:
			# 	end = end - index

			# Change the filter output for this layer as required
			if end != 0: # Concatenate feature maps from previous layer and index+start layer
				filters = output_filters[index + start] + output_filters[end]
			else: # Output the feature map from this layer
				filters = output_filters[index + start]

		elif block['type'] == 'shortcut': # Skip connection like the ones in RESNET models
			shortcut = EmptyLayer()
			module.add_module("shortcut{}".format(index), shortcut)

		elif block['type'] == 'yolo':
			# For choosing which bounding box to use at a specific scale
			mask = block['mask'].split(',')
			mask = [int(x) for x in mask ]

			# Each cell is responsible for predicting 3 bounding boxes
			anchors = block['anchors'].split(',')
			anchors = [int(x) for x in anchors]
			anchors = [(anchors[x], anchors[x+1]) for x in range(0, len(anchors),2)]
			anchors = [x for index, x in enumerate(anchors) if index in mask] # anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors)
			module.add_module("Detection_{}".format(index), detection)

		module_list.append(module)
		output_filters.append(filters)
		prev_filters = filters

	return net_info, module_list


blocks = parse_cfg('cfg/yolov3.cfg')
a = create_modules(blocks)
b = Darknet()
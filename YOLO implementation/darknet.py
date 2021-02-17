#!/usr/bin/env python3
# Based on the tutorial series by Ayoosh Kathuria 
# @ https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
	'''
	Takes a configuration file

	Returns a list of blocks. Each blocks describes a block in the neural 
	network to be built. Block is represented as a dictionary in the list

	Note how the first item of the dictionary, blocks, is going to be
	the network information
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
	the nn.Module module
	'''
	net_info = blocks[0] # Captures the info about the network (input and pre-processing)
	module_list = nn.ModuleList() # Store the block's parameters as a member of the nn.Module module
	prev_filters = 3 # Keeps track of the previous filter depth as the input for the next conv layer. Initialised
					 # as 3 because the initial input image is RGB and has depth 3
	output_filters = [] # Keeps track of the filter depth for each block in case the route laye brings in a 
						# concatenated feature map from previous layers
	for index, block in enumerate(blocks[1:]): # Iterate through the each layer of the network, not including the networks info
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
				activation_fn = nn.LeakyRelu(0.1, inplace=True)
				module.add_moduel("leaky_{}".format(index), activation_fn)

		elif block['type'] == 'upsample' # for upsampling, use Bilinear2dUpsampling:

		print(module)
blocks = parse_cfg('cfg/yolov3.cfg')
create_modules(blocks)
print(blocks[1:5])
print("\n")
print(blocks[0:5])
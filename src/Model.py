from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.ops.rnn import dynamic_rnn

from google.colab import files

# authenticate drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

def ln(tensor, scope=None, epsilon=1e-5):
	##Layer normalizes a 2D tensor along its second axis

	assert (len(tensor.get_shape())==2)
	m, v = tf.nn.moments(tensor, [1], keep_dims=True)
	if not isinstance(scope,str):
		scope=''
	with tf.variable_scope(scope+'layer_norm'): 

		scale= tf.get_variable('scale',shape=[tensor.get_shape()[1]],initializer=tf.constant_initializer(1))
		shift = tf.get_variable('shift',shape=[tensor.get_shape()[1]],initializer=tf.constant_initializer(0))

	ln_initial = (tensor-m)/tf.sqrt(v+epsilon)

	return ln_initial*scale+shift

class MultiDimensionalLSTMCell(RNNCell):

	def __init__(self,num_units,forget_bias=0.0,activation=tf.nn.tanh):
		self._num_units=num_units
		self._forget_bias=forget_bias
		self._activation=activation


	def state_size(self):
		return LSTMStateTuple(self._num_units, self._num_units)

	def output_size(self):
		return self._num_units

	def __call__(self,inputs,state,scope=None):
		'''LSTM Cell@param inputs (batch,n)@param state : the states and hidden units of the two cells'''

		with tf.variable_scope(scope or type(self).__name__):
			c1,c2,h1,h2 = state

		#change bias argument to False since LN will add bias via shift
		self._num_units = self.output_size()

		concat = _linear([inputs, h1, h2], 5 * self._num_units, False)
		i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

		# add  layer normalization to each gate

		i = ln(i, scope='i/')
		j = ln(j, scope='j/')
		f1 = ln(f1, scope='f1/')
		f2 = ln(f2, scope='f2/')
		o = ln(o,scope='o/')

		new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *self._activation(j))

		# add layer_normalization in calculation of new hidden state

		new_h = self._activation(ln(new_c,scope='new_h/'))*tf.nn.sigmoid(o)
		new_state = LSTMStateTuple(new_c,new_h)

		return new_h, new_state

def multi_dimensional_rnn_while_loop(rnn_size, input_data, sh, dims=None, scope_n="layer1"):
    """Implements naive multi dimension recurrent neural networks
    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param sh: [height,width] of the windows
    @param dims: dimensions to reverse the input data,eg.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope
    returns [batch,h/sh[0],w/sh[1],rnn_size] the output of the lstm
    """

    with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):

        # Create multidimensional cell with selected size
        cell = MultiDimensionalLSTMCell(rnn_size)

        # Get the shape of the input (batch_size, x, y, channels)
        batch_size, X_dim, Y_dim, channels = input_data.shape.as_list()
        # Window size
        X_win, Y_win = sh
        # Get the runtime batch size
        batch_size_runtime = tf.shape(input_data)[0]

        # If the input cannot be exactly sampled by the window, we patch it with zeros
        if X_dim % X_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_win - (X_dim % X_win), Y_dim, channels])
            # Concatenate X dimension
            input_data = tf.concat(axis=1, values=[input_data, offset])
            # Update shape value
            X_dim = input_data.shape[1].value

        # The same but for Y axis
        if Y_dim % Y_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_dim, Y_win - (Y_dim % Y_win), channels])
            # Concatenate Y dimension
            input_data = tf.concat(axis=2, values=[input_data, offset])
            # Update shape value
            Y_dim = input_data.shape[2].value

        # Get the steps to perform in X and Y axis
        h, w = int(X_dim / X_win), int(Y_dim / Y_win)

        # Get the number of features (total number of input values per step)
        features = Y_win * X_win * channels

        # Reshape input data to a tensor containing the step indexes and features inputs
        # The batch size is inferred from the tensor size
        x = tf.reshape(input_data, [batch_size_runtime, h, w, features])

        # Reverse the selected dimensions
        if dims is not None:
            assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)

        # Reorder inputs to (h, w, batch_size, features)
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (h*w*batch_size , features)
        x = tf.reshape(x, [-1, features])
        # Split tensor into h*w tensors of size (batch_size , features)
        x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

        # Create an input tensor array (literally an array of tensors) to use inside the loop
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
        # Unstack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        # Create an input tensor array for the states
        states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
        # And an other for the output
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')

        # initial cell hidden states
        # Write to the last position of the array, the LSTMStateTuple filled with zeros
        states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size_runtime, rnn_size], tf.float32),
                                                          tf.zeros([batch_size_runtime, rnn_size], tf.float32)))

        # Function to get the sample skipping one row
        def get_up(t_, w_):
            return t_ - tf.constant(w_)

        # Function to get the previous sample
        def get_last(t_, w_):
            return t_ - tf.constant(1)

        # Controls the initial index
        time = tf.constant(0)
        zero = tf.constant(0)

        # Body of the while loop operation that applies the MD LSTM
        def body(time_, outputs_ta_, states_ta_):

            # If the current position is less or equal than the width, we are in the first row
            # and we need to read the zero state we added in row (h*w). 
            # If not, get the sample located at a width distance.
            state_up = tf.cond(tf.less_equal(time_, tf.constant(w)),
                               lambda: states_ta_.read(h * w),
                               lambda: states_ta_.read(get_up(time_, w)))

            # If it is the first step we read the zero state if not we read the immediate last
            state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
                                 lambda: states_ta_.read(get_last(time_, w)),
                                 lambda: states_ta_.read(h * w))

            # We build the input state in both dimensions
            current_state = state_up[0], state_last[0], state_up[1], state_last[1]
            # Now we calculate the output state and the cell output
            out, state = cell(inputs_ta.read(time_), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(time_, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(time_, state)

            # Return outputs and incremented time step 
            return time_ + 1, outputs_ta_, states_ta_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, tf.constant(h * w))

        # Run the looped operation
        result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                      parallel_iterations=1)

        # Extract the output tensors from the processesed tensor array
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # Reshape outputs to match the shape of the input
        y = tf.reshape(outputs, [h, w, batch_size_runtime, rnn_size])

        # Reorder te dimensions to match the input
        y = tf.transpose(y, [2, 0, 1, 3])
        # Reverse if selected
        if dims is not None:
            y = tf.reverse(y, dims)

        # Return the output and the inner states
        print("Implementing the 2D LSTM")
        return y, states


class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen =32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
		"init model: add CNN, RNN and CTC and initialize TF"
		self.dump = dump
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0
		# Whether to use normalization over a batch or a population
		self.is_train = tf.placeholder(tf.bool, name='is_train')

		# input image batch
		self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

		# setup CNN, RNN and CTC
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()

		# setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		# initialize TF
		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self):
		"create CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(self.inputImgs, axis=3)

		# list of parameters for the layers
		kernelVals = [5, 5, 3, 3, 3]
		kernelVals = [7, 3, 3, 3, 3, 3, 3, 3]		
		featureVals = [1, 32, 64, 128, 128, 256]
		featureVals = [1, 32, 64, 64, 64, 128, 128, 512, 512]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		strideVals = poolVals = [(2,2), (1,2), (1,2), (1,2), (1,1), (1,1), (1,1), (2,2) ]
		numLayers = len(strideVals)

		# create layers
		# input to first CNN layer

		pool = cnnIn4d

		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
			pool2 = tf.contrib.layers.fully_connected(pool,128)
			pool3 = tf.contrib.layers.fully_connected(pool2,512)
			#print("Pool 2's shape : ")
			#rint(pool2.shape)

		self.cnnOut4d = pool3
		


	def setupRNN(self):
		"create RNN layers and return output of these layers"
		#print(self.cnnOut4d.shape)
		rnnIn3d = self.cnnOut4d
		#rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

		# basic cells which is used to build RNN
		numHidden = 256
		#cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		# stack basic cells
		#stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		#((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
		
		## Implementing the 2D LSTM
		rnn_out,_ = multi_dimensional_rnn_while_loop(rnn_size=numHidden, input_data=rnnIn3d, sh=[1,1])


		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		#concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		#numHidden used to be numHidden*2 in the arguments of the following line
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden, len(self.charList) + 1], stddev=0.1))
		##rnn_out used to be "concat" in the following line's arguments
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=rnn_out, filters=kernel, rate=1, padding='SAME'), axis=[2])
		#print(self.rnnOut3d)

	def setupCTC(self):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC
		#print(self.rnnOut3d.shape)
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1,0,2])
		#self.ctcIn3dTBC =tf.reshape(self.ctcIn3dTBC,[32,batchSize,80])
		#print(self.ctcIn3dTBC.shape)
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		# calc loss for each element to compute label probability
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

			print(word_beam_search_module)
			# prepare information about language (dictionary, characters in dataset, characters forming words) 
			chars = str().join(self.charList)
			### These things not required!############ as we don't use wordbeamsearch#######
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()
			##########
			print(dir(word_beam_search_module))

			# decode using the "Words" mode of word beam search
			self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))
			
			res = sess.run(self.decoder, {tf.nn.softmax(self.ctcIn3dTBC, dim=2): feedMat})

			# feed matrix of shape TxBxC and evaluate TF graph
			#res = sess.run(decode, {mat: feedMat})
			

	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)

		sess=tf.Session() # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)


	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal


	def dumpNNOutput(self, rnnOutput):
		"dump the output of the NN to CSV file(s)"
		dumpDir = '../dump/'
		if not os.path.isdir(dumpDir):
			os.mkdir(dumpDir)

		# iterate over all batch elements and create a CSV file for each one
		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			fn = dumpDir + 'rnnOutput_'+str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)
			with open(fn, 'w') as f:
				f.write(csv)


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		"feed a batch into the NN to recognize the texts"
		
		# decode, optionally save RNN output
		numBatchElements = len(batch.imgs)
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)

		# dump the output of the NN to CSV file(s)
		if self.dump:
			self.dumpNNOutput(evalRes[1])

		return (texts, probs)
	

	def save(self):

		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
		# Authenticate and create the PyDrive client.
		# This only needs to be done once in a notebook.
		auth.authenticate_user()
		gauth = GoogleAuth()
		gauth.credentials = GoogleCredentials.get_application_default()
		drive = GoogleDrive(gauth)
		print("Deleting old snap")

		for old_file in os.listdir('/content/drive/My Drive/'):
			if ('checkpoint' == old_file or 'snap' in old_file or
				'accuracy' in old_file):
				os.remove('/content/drive/My Drive/' + old_file)

		print("Deleted old snap. Uploading new snap")

		for file in os.listdir(r'/content/OCR_model/model'):
			if 'checkpoint' == file or 'snap' in file or 'accuracy' in file:
				uploaded = drive.CreateFile({'title': file})
				uploaded.SetContentFile(r'/content/OCR_model/model/' + file)
				uploaded.Upload()
				print('Uploaded file with ID {}'.format(uploaded.get('id')))


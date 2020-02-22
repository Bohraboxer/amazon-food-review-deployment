import packages_to_import as p
from keras import Input, Model

from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Embedding,LSTM,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Bidirectional



class model_():
	def __init__(self,max_length,neuron1,neuron2,dropout1,dropout2,vocab_size):
		self.max_length=max_length
		self.neuron1=neuron1
		self.neuron2=neuron2
		self.dropout1=dropout1
		self.dropout2=dropout2
		self.vocab_size=vocab_size

	def model_fitting(self):
		input=Input((self.max_length,))
		embedding=Embedding(self.vocab_size,32,input_length=self.max_length)(input)
		x=Bidirectional(LSTM(self.neuron1,return_sequences=True))(embedding)
		x=Dropout(self.dropout1)(x)
		x=Bidirectional(LSTM(self.neuron2))(x)
		x=Dropout(self.dropout2)(x)
		output=Dense(1,activation='sigmoid')(x)
		model=Model(inputs=input,outputs=output)
		return model




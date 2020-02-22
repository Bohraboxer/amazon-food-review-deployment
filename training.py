import packages_to_import as p
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import p_metric
from p_metric import *
from keras.models import model_from_json
import pickle
from keras.models import load_model
import model
from model import *
from sklearn.utils import class_weight


class train_test():
	def __init__(self,test_data):
		self.test_data=test_data
		# pickle_in=open("review.pickle",'rb')
		# self.data=pickle.load(pickle_in)

		# self.tokenizer=Tokenizer()
		# self.tokenizer.fit_on_texts(self.data['Text'].values)
		# tokenizer_pickle = open("tokenizer.pickle", "wb")
		# pickle.dump(self.tokenizer, tokenizer_pickle)
		# tokenizer_pickle.close()
		self.tokenizer_pickle=open('tokenizer.pickle','rb')
		self.tokenizer=pickle.load(self.tokenizer_pickle)





	def train(self):
		train_post_seq=self.tokenizer.texts_to_sequences(self.data['Text'].values)
		train_post_seq_padded=pad_sequences(train_post_seq,maxlen=125, padding='post')
		X_train,X_test,y_train,y_test=train_test_split(train_post_seq_padded,self.data['Sentiment'],test_size=0.25)
		#max_length,neuron1,neuron2,dropout1,dropout2
		vocab_size=len(self.tokenizer.word_index)+1
		class_weights = class_weight.compute_class_weight('balanced',p.np.unique(y_train),y_train)
		model=model_(125,64,48,0.3,0.25,vocab_size).model_fitting()
		model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',p_metric.f1_m,p_metric.precision_m,p_metric.recall_m])
		print("*"*100)
		print('Training')
		print("*"*100)
		early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
		model.fit(X_train, y_train,batch_size=50,epochs=50,callbacks=[early_stopping],validation_data=(X_test, y_test),class_weight=class_weights)
		scores=model.evaluate(X_test,y_test,verbose=0)
		print("Test score:",scores[0]) 
		print("Test accuracy:",scores[1])
		model_json=model.to_json()
		with open ("model.json",'w') as json_file:
			json_file.write(model_json)
			model.save_weights("model.h5")
			print("Saved model to disk")


	def testing(self):
		stop=p.stopwords.words('english')
		#print("Replacing None values with Nodata")
		self.test_data['Text'] = self.test_data['Text'].astype(str)
		self.test_data['Text']=self.test_data['Text'].apply(lambda x:" ".join(x.lower() for x in x.split()))
		self.test_data['Text'].fillna("nonedata",inplace=True)
		#print("*"*100)
		#print("The data is empty - ",test_data['Text'].isnull().values.any())
		self.test_data['Text'] = self.test_data['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
		self.test_data['Text'] = self.test_data['Text'].str.replace('<[^<]+?>', '')
		self.test_data['Text'] = self.test_data['Text'].str.replace('http\S+|www.\S+', '',case=False)
		self.test_data['Text'] = self.test_data['Text'].str.replace('\d+', '')
		self.test_data['Text'] = self.test_data['Text'].str.replace('[^\w\s]','')
		self.test_data['Text'] = self.test_data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		self.test_data['Text'].apply(lambda x: " ".join([p.Word(word).lemmatize() for word in x.split()]))
		#print("*"*30,"Cleaning Successful","*"*30)
		post_seq_test=self.tokenizer.texts_to_sequences(self.test_data['Text'].values)
		post_seq_padded_test=pad_sequences(post_seq_test,maxlen=125, padding='post')
		json_file=open('model.json','r')
		loaded_model_json=json_file.read()
		json_file.close()
		model=model_from_json(loaded_model_json)
		model.load_weights('model.h5')
		#print("Loaded model from disk")
		predicted=model.predict(post_seq_padded_test)
		return predicted


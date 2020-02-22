import packages_to_import as p

class understanding_data:
	def __init__(self,data):
		self.data=data
		self.stop=p.stopwords.words('english')
		self.freq_for_text=None
		self.freq_for_summary=None
		self.freq_for_text_without_stopwords=None
		self.freq_for_Summary_without_stopwords=None
		# seeing the size of data
		print("Shape of the data is:",self.data.shape)
		print("Column names of the datset are : ",self.data.columns)
		print("Few data samples are :",self.data.head())
		print("#"*50)
		self.data=self.data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first',inplace=False)
		print("After removing duplicate data the shape of the dataset is",self.data.shape)
		
		self.data=self.data[self.data.HelpfulnessNumerator<=self.data.HelpfulnessDenominator]
		print("After removing the invalid data, the shape of the dataset is",self.data.shape)
		print("Text column has null value is stated to be",self.data['Text'].isnull().values.any())
		print("If any of the text row is empty, then it would be filled with None",self.data['Text'].fillna("none",inplace=True))
		print("Summary column has null value is stated to be",self.data['Summary'].isnull().values.any())
		print("If any of the Summary row is empty, then it would be filled with None",self.data['Summary'].fillna("none",inplace=True))
		print("#"*15," For the column in Text ","#"*15)
		self.data['word_count']=data['Text'].apply(lambda x: len(str(x).split(" ")))
		print("Number of words in a review(for first five reviews) is:",self.data['word_count'].head())
		self.data['char_count']=self.data['Text'].str.len()
		print("Number of character count in a review(for first five reviews) is:",self.data['char_count'].head())
		self.data['stopwords']=self.data['Text'].apply(lambda x:len([x for x in x.split() if x in self.stop]))
		print("Number of stopwords in a review(for first five reviews) is:",self.data['stopwords'].head())
		self.data['numeric']=self.data['Text'].apply(lambda x:len([x for x in x.split() if x.isdigit()]))
		print("Number of numeric count in a review(for first five reviews) is:",self.data['numeric'].head())	
		self.data['upper']=self.data['Text'].apply(lambda x:len([x for x in x.split() if x.isupper()]))
		print("Number of uppercase in a review(for first five reviews) is:",self.data['upper'].head())

		print("#"*15," For the column in Summary ","#"*15)
		self.data['Summary_word_count']=data['Summary'].apply(lambda x: len(str(x).split(" ")))
		print("Number of words in a review(for first five Summary) is:",self.data['Summary_word_count'].head())
		self.data['Summary_char_count']=self.data['Summary'].str.len()
		print("Number of character count in a review(for first five Summary) is:",self.data['Summary_char_count'].head())
		self.data['Summary_stopwords']=self.data['Summary'].apply(lambda x:len([x for x in x.split() if x in self.stop]))
		print("Number of stopwords in a review(for first five Summary) is:",self.data['Summary_stopwords'].head())
		self.data['Summary_numeric']=self.data['Summary'].apply(lambda x:len([x for x in x.split() if x.isdigit()]))
		print("Number of numeric count in a review(for first five Summary) is:",self.data['Summary_numeric'].head())
		self.data['Summary_upper']=self.data['Summary'].apply(lambda x:len([x for x in x.split() if x.isupper()]))
		print("Number of uppercase in a review(for first five Summary) is:",self.data['Summary_upper'].head())
		print("*"*100)
		self.freq_for_text = p.pd.Series(' '.join(self.data['Text']).split()).value_counts()
		print("Top 20 most occuring words are",self.freq_for_text[:20])
		self.freq_for_Summary = p.pd.Series(' '.join(self.data['Summary']).split()).value_counts()
		print("Top 20 most occuring words are",self.freq_for_Summary[:20])
		print("*"*100)
		self.data['Text_without_stopword'] = self.data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in self.stop))
		self.data['Summary_without_stopword'] = self.data['Summary'].apply(lambda x: " ".join(x for x in x.split() if x not in self.stop))
		print("*"*100)
		self.freq_for_text_without_stopwords = p.pd.Series(' '.join(self.data['Text_without_stopword']).split()).value_counts()
		print("Top 20 most occuring words after removing stopwords are",self.freq_for_text_without_stopwords[:20])
		self.freq_for_Summary_without_stopwords = p.pd.Series(' '.join(self.data['Summary_without_stopword']).split()).value_counts()
		print("Top 20 most occuring words after removing stopwords are",self.freq_for_Summary_without_stopwords[:20])
		print("*"*100)








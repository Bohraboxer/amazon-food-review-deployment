import packages_to_import as p


def df(data):
	data=data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first',inplace=False)
	data=data[data.HelpfulnessNumerator<=data.HelpfulnessDenominator]
	stop=p.stopwords.words('english')
	print("Replacing None values with Nodata")
	data['Text']=data['Text'].apply(lambda x:" ".join(x.lower() for x in x.split()))
	data['Text'].fillna("nonedata",inplace=True)
	data['Summary'].fillna("nonedata",inplace=True)
	print("*"*100)
	data['Text']=data['Summary'].str.cat(data['Text'],sep=' ')
	print("The data is empty - ",data['Text'].isnull().values.any())
	data['Text'] = data['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
	data['Text'] = data['Text'].str.replace('<[^<]+?>', '')
	data['Text'] = data['Text'].str.replace('http\S+|www.\S+', '',case=False)
	data['Text'] = data['Text'].str.replace('\d+', '')
	data['Text'] = data['Text'].str.replace('[^\w\s]','')
	data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
	data['Text'] = data['Text'].apply(lambda x: " ".join([p.Word(word).lemmatize() for word in x.split()]))
	print("*"*30,"Cleaning Successful","*"*30)
	data['Sentiment'] = p.np.where(data['Score']<3,0,1)
	review_pickle = open("review.pickle","wb")
	p.pickle.dump(data[['Text','Sentiment']], review_pickle)
	review_pickle.close()
	print("Pickled")


	







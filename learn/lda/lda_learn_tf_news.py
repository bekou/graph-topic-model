import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.externals import joblib
import lda


name_vector=['id','type','review']



train_ds=pd.read_csv("../../ds/new_groups/news_groups_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
train_data_unsupervised=pd.read_csv("../../ds/new_groups/news_groups_unsupervised.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()


train_ds, eval_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)


label = preprocessing.LabelEncoder()
train_ds[:,0]= label.fit_transform(train_ds[:,1])
eval_ds[:,0]= label.transform(eval_ds[:,1])

list_topics=[100]
list_features=[5000]
iters_list=[3000]
for i in range(len(list_features)):
    for j in range(len(list_topics)):
	for z in range(len(iters_list)):
		iters=iters_list[z]
		print list_features[i]
		print list_topics[j]
	
	

		vectorizer = CountVectorizer(max_features=list_features[i],stop_words='english')
		train_learn_unigrams=vectorizer.fit_transform(train_data_unsupervised[:,2])
		train_unigrams=vectorizer.transform(train_ds[:,2])
		eval_unigrams=vectorizer.transform(eval_ds[:,2])

		print "start lda"
		model = lda.LDA(n_topics=list_topics[j], n_iter=iters, random_state=1)
		train_learn_lda=model.fit_transform(train_learn_unigrams) # model.fit_transform(X) is also available
		print "saving the object"
		joblib.dump(model, 'objects_lda_learn_news/lda_learn_news_tf_'+str(iters)+'_'+str(list_features[i])+"_"+str(list_topics[j])+".pkl") 
		
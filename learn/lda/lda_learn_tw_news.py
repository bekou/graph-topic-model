import pandas as pd
import codecs
import numpy as np
from sklearn import cross_validation as cv 
import createGraphFeatures as graph
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import lda


cols = ['id','class', 'text']

train_ds=pd.read_csv("../../ds/new_groups/news_groups_train.txt",names=cols,sep="###",encoding="utf-8",engine='python').as_matrix()
train_unsupervised=pd.read_csv("../../ds/new_groups/news_groups_unsupervised.txt",names=cols,sep="###",encoding="utf-8",engine='python')


train_ds, eval_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)

train=pd.DataFrame(train_ds, columns = cols)
eval_set=pd.DataFrame(eval_ds, columns = cols)

y_train=train['class']
y_eval_set=eval_set['class']
sliding_window = 2
b = 0.003

idf_pars = ["no"]

centrality_pars = ["degree_centrality","in_degree_centrality","out_degree_centrality","weighted_centrality"]

list_features=[5000]
for idf_par in idf_pars:
	for i in range(len(list_features)):	
	    for centrality_par in centrality_pars:
		try:
			print "idf:"+idf_par
			print "centrality_par:"+centrality_par
		
			centrality_col_par = "eigenvector_centrality"
			print "centrality_col_par:"+centrality_col_par

			# Get the number of documents based on the dataframe column size
			num_documents = train_unsupervised.shape[0]

			# Initialize an empty list to hold the clean-preprocessed documents
			clean_train_unsuper_documents = []
			unique_words = []
			print "Computing unique words"
			# Loop over each document; create an index i that goes from 0 to the length
			# of the document list 
			clean_train_unsuper_documents=train_unsupervised['text'].tolist()
			vectorizer = CountVectorizer(max_features=list_features[i],stop_words='english')
			vectorizer.fit_transform(train_unsupervised['text'])
			unique_words=vectorizer.get_feature_names()

			print "Unique words:"+str(len(unique_words))

		

			features = graph.createGraphFeatures(num_documents,clean_train_unsuper_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
		
			print "Train unsupervised gow computed"
        	except Exception as e:
			print e
			print "train unsupervised error"
		try:
			## eval_seting set
			
			print train.shape

			# Get the number of documents based on the dataframe column size
			num_train_documents = train.shape[0]

			# Initialize an empty list to hold the clean-preprocessed documents
			clean_train_documents = []
			clean_train_documents=train['text'].tolist()
		
			print "Computing train gow"
			train_features =  graph.createGraphFeatures(num_train_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
			print "Train  gow computed"
		
			print train_features.shape
		except Exception as e:
			print e
			print "train gow error"
		try:
			print eval_set.shape

			# Get the number of documents based on the dataframe column size
			num_eval_documents = eval_set.shape[0]

			# Initialize an empty list to hold the clean-preprocessed documents
			clean_eval_documents = []
			clean_eval_documents=eval_set['text'].tolist()

			print "Computing eval_set gow"
			eval_features =  graph.createGraphFeatures(num_eval_documents,clean_eval_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
			print eval_features.shape
			print "eval_set gow computed"

		except Exception as e:
			print e
			print "eval_set gow error"
		try:

			label = preprocessing.LabelEncoder()
			y_train= label.fit_transform(y_train)
			y_eval_set= label.transform(y_eval_set)


			
			list_topics=[100]

			
			list_iterations=[3000]
			
			
		except Exception as e:
			print e
			print "Preprocessing error"

		for z in range(len(list_iterations)):
		
				    for j in range(len(list_topics)):
					print list_features[i]
					print list_topics[j]
					print list_iterations[z]
					iters=list_iterations[z]
					n_topics = list_topics[j]
					try:
	
						print("convert text into sparse matrix...")

						#vectorizer = CountVectorizer(max_features=list_features[i],stop_words='english')
				
						model = lda.LDA(n_topics=list_topics[j], n_iter=iters, random_state=1)
						train_learn_lda=model.fit_transform(features)
					
		
						print "Saving the object"
						joblib.dump(model, 'objects_lda_learn_news/lda_learn_news_tw_'+str(centrality_par)+"_"+str(sliding_window)+"_"+str(iters)+'_'+str(list_features[i])+"_"+str(list_topics[j])+".pkl") 
						print 'objects_lda_learn_news/lda_learn_news_tw_'+str(centrality_par)+"_"+str(sliding_window)+"_"+str(iters)+'_'+str(list_features[i])+"_"+str(list_topics[j])+".pkl"
						print "transforming train and eval_set"
					except Exception as e:
			            		print e
						print "Error while computing learn features"
					
					



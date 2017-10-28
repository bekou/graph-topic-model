import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import lda
from sklearn.externals import joblib
import createGraphFeatures as graph
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score


def printAccuracies(data_train, data_test ,lab_col,dataset):


	
	sliding_window =2
	b = 0.003
	
	idf_pars = ["no"]

	#sliding_windows=[2,3,4]
	centrality_pars = ["degree_centrality"]


	y_train=data_train['class']
	y_test=data_test['class']
	train_reviews=data_train['text']
	test_reviews=data_test['text']
	idf_par='no'
	label = preprocessing.LabelEncoder()
	train_labels= label.fit_transform(y_train)
	test_labels= label.transform(y_test)


	

	
	#for sliding_window in sliding_windows
	for centrality_par in centrality_pars:
	
	
				
		train_unsupervised=pd.read_csv("../../ds/new_groups/news_groups_unsupervised.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')

		
		list_features=[5000]
		comp=[1187]
		alphas=[0.00001]
		normal="no"


		if normal=="no":
			idf_par="no"
			normalized_centrality=False
			folder_loc="tw/"+str(sliding_window)+"/lsi_learn_tw_"
		elif normal=="yes":
			idf_par="no"
			normalized_centrality=True
			folder_loc="tw_norm/lsi_learn_tw_norm_"
		elif normal=="idf":
			idf_par="idf"
			normalized_centrality=True
			folder_loc="tw_idf/lsi_learn_tw_idf_"
	
	
	

		
		label = preprocessing.LabelEncoder()
		train_labels= label.fit_transform(y_train)
		test_labels= label.transform(y_test)
		

		for j in range(len(list_features)):
			for i in range(len(comp)):
				
					try:
						
						max_features_nr=list_features[j]
						
						model=joblib.load('../../learn/lsi/objects_lsi_learn_news/'+folder_loc+str(centrality_par)+"_"+str(sliding_window)+"_"+str(comp[i])+'_'+str(list_features[j])+".pkl")
						
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
								vectorizer = CountVectorizer(max_features=max_features_nr,stop_words='english')
								vectorizer.fit_transform(train_unsupervised['text'])
								unique_words=vectorizer.get_feature_names()

								print "Unique words:"+str(len(unique_words))		

								#features = graph.createGraphFeatures(num_documents,clean_train_unsuper_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par).astype(int)
		
								print "Train unsupervised gow computed"
						except Exception as e:
								print e
								print "train unsupervised error"



						try:

								# Get the number of documents based on the dataframe column size
								num_documents =data_train.shape[0]

								# Initialize an empty list to hold the clean-preprocessed documents
								clean_train_documents = []
							
							
								# Loop over each document; create an index i that goes from 0 to the length
								# of the document list 
								clean_train_documents=data_train['text'].tolist()							

								print "Unique words:"+str(len(unique_words))
			
								print "Computing train gow"
								if normal=="no":
									train_features = graph.createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality).astype(int)
								else:
									train_features = graph.createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality)
		


						except Exception as e:
								print e
								print "train  error"
						try:
			
								print data_test.shape

								# Get the number of documents based on the dataframe column size
								num_test_documents = data_test.shape[0]

								# Initialize an empty list to hold the clean-preprocessed documents
								clean_test_documents = []
								clean_test_documents=data_test['text'].tolist()
			
								print "Computing test gow"

								if normal=="no":
									test_features =  graph.createGraphFeatures(num_test_documents,clean_test_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality).astype(int)
								else:
									test_features =  graph.createGraphFeatures(num_test_documents,clean_test_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality)
				

								print test_features.shape
						except Exception as e:
								print e
								print "test  error"



						
	

						train_lsi=model.transform(train_features) # model.fit_transform(X) is also available

						test_lsi=model.transform(test_features) # model.fit_transform(X) is also available


						for z in range(len(alphas)):
							clf = SGDClassifier(alpha=alphas[z],random_state=42)
							clf.fit(train_lsi, np.array(train_labels).astype(int))     
							ypred = clf.predict(test_lsi)
							print "The  accuracy of SVM-linear: "+ str(accuracy_score(test_labels.astype(int), ypred))

							acc=accuracy_score(test_labels.astype(int), ypred)
							macro=precision_recall_fscore_support(test_labels.astype(int), ypred, average='macro')
							micro=precision_recall_fscore_support(test_labels.astype(int), ypred, average='micro')
							macrof1=f1_score(test_labels.astype(int), ypred, average='macro')  
							microf1=f1_score(test_labels.astype(int), ypred, average='micro')  
							weightedf1=f1_score(test_labels.astype(int), ypred, average='weighted')
							nonef1=f1_score(test_labels.astype(int), ypred, average=None)		

								
							print  acc
							with codecs.open("gow_ds_accuracy_all_no_norm_news.txt", "a", "utf-8") as myfile:
									      myfile.write(dataset+"\t"+normal+"\t"+str(alphas[z])+"\t"+str(list_features[j])+"\t"+str(comp[i])+'\t' + "gow\t"+str(centrality_par)+'\t'+str(sliding_window)+ '\t' +str(acc)+'\t'+str(macro[0])+'\t'+str(macro[1])+'\t'+str(micro[0])+'\t'+str(micro[1])+'\t'+str(macrof1)+'\t'+str(microf1)+'\t'+str(weightedf1)+'\t'+str(nonef1[0])+'\t'+str(nonef1[1])+'\t' +"\n")
										
						

						

					except Exception as e:
							print e
							print "Exception"+"\t"+dataset+ "LDA"+"\t"+str(comp[i])+"\t"+str(max_features_nr)
		
		

name_vector=['id','class','text']

train_ds=pd.read_csv("../../ds/new_groups/news_groups_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')
test_ds=pd.read_csv("../../ds/new_groups/news_groups_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')


printAccuracies(train_ds,test_ds,1,"news_groups")

train_ds=pd.read_csv("../../ds/R8/r8_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')
test_ds=pd.read_csv("../../ds/R8/r8_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')


printAccuracies(train_ds,test_ds,1,"R8")

train_ds=pd.read_csv("../../ds/R52/r52_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')
test_ds=pd.read_csv("../../ds/R52/r52_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')


printAccuracies(train_ds,test_ds,1,"R52")



train_ds=pd.read_csv("../../ds/bbc_news/bbc_news.txt",names=name_vector,sep="###",encoding="utf-8",engine='python')
train_ds, test_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)


printAccuracies(train_ds,test_ds,1,"bbc_news")


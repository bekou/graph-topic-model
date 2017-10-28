import pandas as pd
import codecs
import numpy as np
from sklearn import cross_validation as cv 
import createGraphFeatures as graph
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.lda import LDA
import lda
import lda.datasets
from sklearn.decomposition import PCA


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

#idf_pars = ["no"]
#normalized_centrality=False

centrality_pars = ["degree_centrality","in_degree_centrality","out_degree_centrality","weighted_centrality"]
list_features=[5000]

normalization=["no"]
for k in range(len(normalization)):
	normal=normalization[k]
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

		
			if normal=="no":
				features = graph.createGraphFeatures(num_documents,clean_train_unsuper_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality).astype(int)
			else:
				features = graph.createGraphFeatures(num_documents,clean_train_unsuper_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality)
		
			print "Train unsupervised gow computed"
		except Exception as e:
			print e
			print "train unsupervised error"
		try:
			## eval_seting set
			#cols2=['id','text']
			#eval_set = pd.read_csv("eval_sample.csv", sep=";", header=0, names=cols2)

			print train.shape

			# Get the number of documents based on the dataframe column size
			num_train_documents = train.shape[0]

			# Initialize an empty list to hold the clean-preprocessed documents
			clean_train_documents = []
			clean_train_documents=train['text'].tolist()
			# Loop over each document; create an index i that goes from 0 to the length
			# of the document list 
			#for i in xrange( 0, num_eval_documents ):
			    # Call our function for each one, and add the result to the list of
			    # clean reviews
			#    clean_eval_documents.append( eval_set['text'][i] )
			    # print train['text'][i]+'\n'
			print "Computing train gow"
			if normal=="no":
				train_features =  graph.createGraphFeatures(num_train_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality).astype(int)
			else:
				train_features =  graph.createGraphFeatures(num_train_documents,clean_train_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality)
			
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
			# Loop over each document; create an index i that goes from 0 to the length
			# of the document list 
			#for i in xrange( 0, num_eval_documents ):
			    # Call our function for each one, and add the result to the list of
			    # clean reviews
			#    clean_eval_documents.append( eval_set['text'][i] )
			    # print train['text'][i]+'\n'
			print "Computing eval_set gow"
			if normal=="no":
				eval_features =  graph.createGraphFeatures(num_eval_documents,clean_eval_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality).astype(int)
			else:
				eval_features =  graph.createGraphFeatures(num_eval_documents,clean_eval_documents,unique_words,sliding_window,b,idf_par,centrality_par,centrality_col_par,normalized_centrality)
						


			print eval_features.shape
			print "eval_set gow computed"

		except Exception as e:
			print e
			print "eval_set gow error"
		try:

			label = preprocessing.LabelEncoder()
			y_train= label.fit_transform(y_train)
			y_eval_set= label.transform(y_eval_set)


			
		except Exception as e:
			print e
			print "Preprocessing error"
		

		#comp=np.linspace(100, train_learn_unigrams.shape[1]-1, num=5).astype(int)
		#comp=np.arange(500,list_features[j],500)
		#print comp
		#for i in range(len(comp)):
		try:
					print "starting PCA"
					pca = PCA(n_components=0.86)
					print "transform to array"
					features=features
					print "fit to array"
					pca.fit(features)
					print "Ratio PCA:"+str(pca.explained_variance_ratio_.sum())+" "+ str(len(pca.explained_variance_ratio_))
					comp=len(pca.explained_variance_ratio_)
					lsa=TruncatedSVD(n_components=comp,random_state=42)
					train_learn_lsa=lsa.fit_transform(features)
					print "saving the object"



					joblib.dump(lsa, 'objects_lsi_learn_news/'+folder_loc+str(centrality_par)+"_"+str(sliding_window)+"_"+str(comp)+'_'+str(list_features[i])+".pkl")

					print("Information kept : "+ str(lsa.explained_variance_ratio_.sum()))

					train_lsa=lsa.transform(train_features)
					eval_lsa=lsa.transform(eval_features)
		
					alphas=[10e-2,10e-3,10e-4,10e-5,10e-6,10e-7,10e-8,10e-9]
					for z in range(len(alphas)):
								clf = SGDClassifier(alpha=alphas[z],random_state=42)
								clf.fit(train_lsa, np.array(y_train).astype(int))     
								ypred = clf.predict(eval_lsa)
								print "The  accuracy of SVM-linear: "+ str(accuracy_score(y_eval_set, ypred))

								acc=accuracy_score(y_eval_set, ypred)
								macro=precision_recall_fscore_support(y_eval_set, ypred, average='macro')
						    		micro=precision_recall_fscore_support(y_eval_set, ypred, average='micro')
								macrof1=f1_score(y_eval_set, ypred, average='macro')  
								microf1=f1_score(y_eval_set, ypred, average='micro')  
								weightedf1=f1_score(y_eval_set, ypred, average='weighted')
								nonef1=f1_score(y_eval_set, ypred, average=None)		

								
								print  acc
								with codecs.open("tw_learn_lsi_eval_results_news.txt", "a", "utf-8") as myfile:
								      myfile.write("newsgroups"+"\t"+normal+"\t"+str(alphas[z])+"\t"+str(list_features[i])+"\t"+str(comp)+'\t' + "gow\t"+str(centrality_par)+'\t'+str(sliding_window)+ '\t' +str(acc)+'\t'+str(macro[0])+'\t'+str(macro[1])+'\t'+str(micro[0])+'\t'+str(micro[1])+'\t'+str(macrof1)+'\t'+str(microf1)+'\t'+str(weightedf1)+'\t'+str(nonef1[0])+'\t'+str(nonef1[1])+'\t' +"\n")
		except Exception as e:
					print e
					print "Exception"+"\t"+str(list_features[i])+ "LSI"+"\t"+str(comp)
					#raise


		


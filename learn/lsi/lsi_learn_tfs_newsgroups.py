import os
import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import lda


name_vector=['id','type','review']


train_ds=pd.read_csv("../../ds/new_groups/news_groups_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
train_data_unsupervised=pd.read_csv("../../ds/new_groups/news_groups_unsupervised.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()

train_ds, eval_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)


label = preprocessing.LabelEncoder()
train_ds[:,0]= label.fit_transform(train_ds[:,1])
eval_ds[:,0]= label.transform(eval_ds[:,1])


#LSI

#list_features=[3000,5000]
list_features=[5000]
normalization=["no"]
for k in range(len(normalization)):
	normal=normalization[k]

	for j in range(len(list_features)):
		if normal=="no":
			vectorizer = CountVectorizer(max_features=list_features[j],stop_words='english')
			folder_loc="tf/lsi_learn_tf_"

		train_learn_unigrams=vectorizer.fit_transform(train_data_unsupervised[:,2])
		train_unigrams=vectorizer.transform(train_ds[:,2])
		eval_unigrams=vectorizer.transform(eval_ds[:,2])
	

		try:
					print "starting PCA"
					pca = PCA(n_components=0.86)
					print "transform to array"
					train_learn_unigrams=train_learn_unigrams.toarray()
					print "fit to array"
					pca.fit(train_learn_unigrams)
					print "Ratio PCA:"+str(pca.explained_variance_ratio_.sum())+" "+ str(len(pca.explained_variance_ratio_))
					comp=len(pca.explained_variance_ratio_)
					lsa=TruncatedSVD(n_components=comp,random_state=42)
					train_learn_lsa=lsa.fit_transform(train_learn_unigrams)
					print "LSI ratio"+str(lsa.explained_variance_ratio_.sum())
					variance=lsa.explained_variance_ratio_.sum()
					print "saving the object"


					#if (lsa.explained_variance_ratio_.sum()>0.85):
					#	
					joblib.dump(lsa, 'objects_lsi_learn_news/'+folder_loc+str(comp)+'_'+str(list_features[j])+".pkl") 


					print("Information kept : "+ str(lsa.explained_variance_ratio_.sum()))


		except Exception as e:
					print e
					print "Exception"+"\t"+str(list_features[j])+ "LSI"+"\t"+str(comp) + "in the transformation or in the saving "

				
		try:	
					if (lsa.explained_variance_ratio_.sum()>0.85):
										train_lsa=lsa.transform(train_unigrams)
										eval_lsa=lsa.transform(eval_unigrams)
										alphas=[10e-2,10e-3,10e-4,10e-5,10e-6,10e-7]

										for z in range(len(alphas)):
												clf = SGDClassifier(alpha=alphas[z],random_state=42)
												clf.fit(train_lsa, np.array(train_ds[:,0]).astype(int))     
												ypred = clf.predict(eval_lsa)
												print "The  accuracy of SVM-linear: "+ str(accuracy_score(eval_ds[:,0].astype(int), ypred))
												
												acc=accuracy_score(eval_ds[:,0].astype(int), ypred)
												macro=precision_recall_fscore_support(eval_ds[:,0].astype(int), ypred, average='macro')
												micro=precision_recall_fscore_support(eval_ds[:,0].astype(int), ypred, average='micro')
												macrof1=f1_score(eval_ds[:,0].astype(int), ypred, average='macro')  
												microf1=f1_score(eval_ds[:,0].astype(int), ypred, average='micro')  
												weightedf1=f1_score(eval_ds[:,0].astype(int), ypred, average='weighted')
												nonef1=f1_score(eval_ds[:,0].astype(int), ypred, average=None)		
												
												
												print  acc
												with codecs.open("tf_learn_lsi_eval_results_news.txt", "a", "utf-8") as myfile:
														myfile.write("news"+'\t'+normal+'\t'+str(alphas[z])+'\t'+str(list_features[j])+'\t'+str(comp)+'\t'+str(variance)+'\t' +str(acc)+'\t'+str(macro[0])+'\t'+str(macro[1])+'\t'+str(micro[0])+'\t'+str(micro[1])+'\t'+str(macrof1)+'\t'+str(microf1)+'\t'+str(weightedf1)+'\t'+str(nonef1[0])+'\t'+str(nonef1[1])+'\t' +"\n")
												
												
		except Exception as e:
					print e
					print "Exception"+"\t"+str(list_features[j])+ "LSI"+"\t"+str(comp)+ "in the classifier "
				

				




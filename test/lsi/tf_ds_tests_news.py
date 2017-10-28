import codecs
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,f1_score
import lda
from sklearn.externals import joblib



def printAccuracies(data_train, data_test,lab_col,dataset):
	
	train_labels=data_train[:,lab_col]
	train_reviews=data_train[:,2]
	test_labels=data_test[:,lab_col]
	test_reviews=data_test[:,2]
	label = preprocessing.LabelEncoder()
	train_labels= label.fit_transform(train_labels)
	test_labels= label.transform(test_labels)
    	

			
	train_data_unsupervised=pd.read_csv("../../ds/new_groups/news_groups_unsupervised.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()



		
	list_features=[5000]
	comp=[659]
	
	alphas=[0.001]

	normal="no"

	
	if normal=="no":
			vectorizer = CountVectorizer(max_features=list_features[0],stop_words='english')
			folder_loc="tf/lsi_learn_tf_"



	for i in range(len(comp)):
		for j in range(len(list_features)):
			
				try:
					
					max_features_nr=list_features[j]
					
					model=joblib.load('../../learn/lsi/objects_lsi_learn_news/'+folder_loc+str(comp[i])+'_'+str(list_features[j])+".pkl")

					
					train_learn_unigrams_lsi=vectorizer.fit_transform(train_data_unsupervised[:,2])
					train_unigrams_lsi=vectorizer.transform(train_reviews)
					test_unigrams_lsi=vectorizer.transform(test_reviews)

	

					train_lsi=model.transform(train_unigrams_lsi) # model.fit_transform(X) is also available

					test_lsi=model.transform(test_unigrams_lsi) # model.fit_transform(X) is also available


			

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
							#printTofile(dataset,"SVM.SVC","linearSVM",'-',"countVectorizer","binary=False"+"\t"+str(len(vectorizer.get_feature_names())),"LSI",str(comp[i]),"-",str(accuracy_score(test_labels.astype(int), ypred))+"\t"+str(lsa.explained_variance_ratio_.sum()))
						print  acc
						with codecs.open("tf_lsi_final_results_news.txt", "a", "utf-8") as myfile:
							      myfile.write(dataset+'\t'+"LSI"+'\t'+normal+'\t'+str(alphas[z])+'\t'+str(list_features[j])+'\t'+str(comp[i])+'\t' +str(acc)+'\t'+str(macro[0])+'\t'+str(macro[1])+'\t'+str(micro[0])+'\t'+str(micro[1])+'\t'+str(macrof1)+'\t'+str(microf1)+'\t'+str(weightedf1)+'\t'+str(nonef1[0])+'\t'+str(nonef1[1])+'\t' +"\n")
						
	
				except Exception as e:
					print e
					print "Exception"+"\t"+str(list_features[j])+ "LSI"+"\t"+str(comp[i])+ "in the classifier "

name_vector=['id','type','review']


train_ds=pd.read_csv("../../ds/new_groups/news_groups_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
test_ds=pd.read_csv("../../ds/new_groups/news_groups_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()


printAccuracies(train_ds,test_ds,1,"news_groups")

train_ds=pd.read_csv("../../ds/R8/r8_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
test_ds=pd.read_csv("../../ds/R8/r8_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()


printAccuracies(train_ds,test_ds,1,"R8")

train_ds=pd.read_csv("../../ds/R52/r52_train.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
test_ds=pd.read_csv("../../ds/R52/r52_test.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()

printAccuracies(train_ds,test_ds,1,"R52")

train_ds=pd.read_csv("../../ds/bbc_news/bbc_news.txt",names=name_vector,sep="###",encoding="utf-8",engine='python').as_matrix()
train_ds, test_ds = cv.train_test_split(train_ds, train_size=0.8, test_size=0.2,random_state=42)

printAccuracies(train_ds,test_ds,1,"bbc_news")




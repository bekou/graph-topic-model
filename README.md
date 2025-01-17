# Graph-based term weighting scheme for topic modeling


This repository contains the code presented in the work:

[Graph-based term weighting scheme for topic modeling](https://bekou.github.io/papers/someris2016/bekoulis-someris2016.pdf)

If you use part of the code please cite:  

```  
@inproceedings{bekoulis2016graph,  
 title={Graph-based term weighting scheme for topic modeling},  
 author={Bekoulis, Giannis and Rousseau, Fran{\c{c}}ois},  
 booktitle={Data Mining Workshops (ICDMW), 2016 IEEE 16th International Conference on Data Mining},  
 pages={1039--1044},  
 year={2016},  
 organization={IEEE}  
}  
```  

### Pre-requisites ###

The code has been developed using Anaconda 2.3 and the LDA module version 1.0.2

Install Anaconda 2.3 from https://repo.continuum.io/archive/ and then 
> pip install lda==1.0.2  


In the learn/lda directory
--------------------------
To train the TF-LDA module:
> python lda_learn_tf_news.py

To train the TW-LDA module:
> python lda_learn_tw_news.py

In the learn/lsi directory
--------------------------
To train the TF-LSI module:
> python lsi_learn_tfs_newsgroups.py

To train the TW-LSI module:
> python lsi_learn_tws_newsgroups.py

In the test/lda directory
--------------------------
To predict using the TF-LDA module:
> python tf_ds_news_group.py

To predict using the TW-LDA module:
> python tw_ds_testsallgow_newsgroup.py

In the test/lsi directory
--------------------------
To predict using the TF-LSI module:
> python tf_ds_tests_news.py

To predict using the TW-LSI module - degree centrality:
> python tw_ds_testsalldegree_gow_news.py

To predict using the TW-LSI module - in-degree centrality:
> python tw_ds_testsallindegree_gow_news.py

To predict using the TW-LSI module - out-degree centrality:
> python tw_ds_testsalloutdegree_gow_news.py

To predict using the TW-LSI module - weighted-degree centrality:
> python tw_ds_testsallweighteddegree_gow_news.py

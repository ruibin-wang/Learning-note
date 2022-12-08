# text classification datasets



## Table shows the detail of this datasets

|Dataset|# Docs|# Words|# Nodes|# Classes|Average Length|Link|
|:----|:----|:----|:----|:----|:----|:----|
|20NG: 20 Newsgroups data set|18,846|42,757|61,603|20|221.26|http://qwone.com/~jason/20Newsgroups/|
|R52|9,100|8,892|17,992|52|69.82|https://www.kaggle.com/datasets/weipengfei/ohr8r52|
|R8|7,674|7,688|15,362|8|65.72|https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection|
|MR|10,662|18,764|29,426|2|20.39|http://www.cs.cornell.edu/people/pabo/movie-review-data/ and https://github.com/mnqu/PTE/tree/master/data/mr|
|MTEB: Massive Text Embedding Benchmark||||||
|Ohsumed corpus|7,400|14,157|21,557|23|135.82|http://disi.unitn.it/moschitti/corpora.htm|







## 20NG

* The 20NG dataset1 (bydate version) contains **18,846 documents** evenly categorized into **20 different categories**. In total, 11,314 documents are in the training set and 7,532 documents are in the test set.


## Ohsumed corpus

* The Ohsumed corpus is from the **MEDLINE database**, which is a bibliographic database of important medical literature maintained by the National Library of Medicine.

* In the work of paper *"Graph Convolutional Networks for Text Classification"*, we used the 13,929 unique cardiovascular diseases abstracts in the first 20,000 abstracts of the year 1991. Each document in the set has one or more associated categories from the 23 disease categories. As we focus on single-label text classification, the documents belonging to multiple categories are excluded so that 7,400 documents belonging to only one category remain. 3,357 documents are in the training set and 4,043 documents are in the test set.


## R52 and R8

* R52 and R8 (all-terms version) are two subsets of the Reuters 21578 dataset. R8 has 8 categories, and was split to 5,485 training and 2,189 test documents. R52 has 52 categories, and was split to 6,532 training and 2,568 test documents.

    * Note: The Reuters-21578 dataset is a collection of documents with news articles. The original corpus has 10,369 documents and a vocabulary of 29,930 words.

## MR

* MR is a **movie review dataset** for binary sentiment classification, in which each review only contains one sentence (Pang and Lee 2005). The corpus has 5,331 positive and 5,331 negative reviews. We used the training/test split in (Tang, Qu, and Mei 2015).



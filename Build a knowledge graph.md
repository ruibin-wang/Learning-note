# Building a knowledge graph

## General KG database

the most popular KG database including: 

|Name of the KG set|Domain|Size|Published Year|Institution|Acessable|Link|
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
|**WordNet**|A Lexical Database for English|117,597 entities, 207,016 facts, 25,229 concepts, 283,070 paris|1995|Princeton Universtiy|Yes|[paper](https://dl.acm.org/doi/pdf/10.1145/219717.219748), [exe_package](https://wordnet.princeton.edu/download/current-version), [web_version](http://wordnetweb.princeton.edu/perl/webwn)|
|**DBpedia**|communitydriven dataset extracted from Wikipedia|∼1,950,000 entities, ∼103,000,000 facts, 259 concepts, 1,900,000 pairs|2007||Yes|[paper](https://link.springer.com/content/pdf/10.1007/978-3-540-76298-0_52.pdf), [dataset](http://wikidata.dbpedia.org/develop/datasets)||
|**YAGO**| general knowledge about people, cities, countries, movies, and organizations|1,056,638 entities and ∼5,000,000 facts, 352,297 concepts, 8,277,227 pairs|updating||Yes|[paper](https://dl.acm.org/doi/pdf/10.1145/1242572.1242667), [dataset](https://yago-knowledge.org/getting-started), [webbroswer](https://yago-knowledge.org/graph)||
|**OpenCyc**|assembled comprehensive ontology and knowledge base|47,000 entities, 306000 facts|2002|W3C Linking Open Data Interest Group|Yes|[paper](https://www.researchgate.net/publication/221250660_An_Introduction_to_the_Syntax_and_Content_of_Cyc), [dataset](https://old.datahub.io/dataset/opencyc)|
|**Freebase**|from wiki knowledge|∼125,000,000 facts, 1,450 concepts, 24,483,434 pairs|2008, 2016 closed|MetaWeb -> Google |Yes|[paper](https://dl.acm.org/doi/pdf/10.1145/1376616.1376746), [dataset](https://developers.google.com/freebase/)|
|**NELL**|use Never-Ending Language Learner to get the knowledge from web|high confidence in 2,810,379 of these beliefs, 	123 concepts, < 242,453 pairs|2010||Yes|[paper](http://rtw.ml.cmu.edu/papers/carlson-aaai10.pdf) ,[dataset](http://rtw.ml.cmu.edu/rtw/)|
|**Wikidata**|structured wikipedia data|14,449,300 entities, 30,263,656 facts|2014||Yes|[paper](https://dl.acm.org/doi/pdf/10.1145/2629489), [datasets](https://www.wikidata.org/wiki)|
|**Google KG**|real-world entities like people, places, and things|> 500 million entities, > 3.5 billion facts||Google|Yes|[API](https://developers.google.com/knowledge-graph)|
|**Probase**|certain general knowledge or certain common sense|	2,653,872 concepts, 20,757,545 pairs||Microsoft|Yes|[paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/05/paper.pdf)[datasets](https://www.microsoft.com/en-us/research/project/probase/)|





### WordNet

* WordNet与一般字典的不同在于组织结构的不同，它是以同义词集合(Synset)作为基本的构建单位来组织的，用户可以在同义词集合中找到一个合适的词去表达一个已知的概念。而与传统词典类似的是它也给出了定义和例句。

* WordNet的词汇结构包括九大类：
    * 上下位关系（动词、名词）
    * 蕴含关系（动词）
    * 相似关系（名词）
    * 成员部分关系（名词）
    * 物质部分关系（名词）
    * 部件部分关系（名词）
    * 致使关系（动词）
    * 相关动词关系（动词）
    * 属性关系（形容词）



* WordNet的英文词条共计：117659条，但中文对应词条却只有42312条。看起来所占比例足足有三分之一，但实际上英文的名词有：82115条，中文的名词则只有两万七千余条


* WordNet中的词性只有四种，即名词、动词、形容词、副词。

* Example:
    ```python
    ## first, download the corpus
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # then, import the packages
    from nltk.corpus import wordnet as wn

    wn.synsets('dog')

    output: 

    ## Synset由三部分组成，第一部分是词义，第二部分是词性，第三部分是编号。
    [Synset('dog.n.01'), Synset('frump.n.01'), Synset('dog.n.03'), Synset('cad.n.01'),Synset('frank.n.02'), Synset('pawl.n.01'), Synset('andiron.n.01'),Synset('chase.v.01')]
    ```


### DBpedia

* DBpedia (from "DB" for "database") is a project aiming to extract structured content from the information created in the Wikipedia project. DBpedia allows users to semantically query relationships and properties of Wikipedia resources, including links to other related datasets.

* the dataset link can be found at: http://wikidata.dbpedia.org/develop/datasets 

* also can be found in the huggingface as a package, named "dbpedia_14". There are totally <font color='red'>**14**</font> labels(categries) in this datasets. For each of the 14 classes we have 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000.
    ```python
    ## install the dataset
    pip install datasets

    ## load dataset using python code
    from datasets import list_datasets, load_dataset

    ## to see how many datasets are included in this package
    print(len(list_datasets()))
    output:
    13348

    ## examples of the dbpedia_14 datasets
    print(load_dataset('dbpedia_14', split='train')[0])
    output:
    {'label': 0, 'title': 'E. D. Abbott Ltd', 'content': ' Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.'} 

    ## another example is 'squad'
    print(load_dataset('squad', split='train')[0])
    output:

    {'id': '5733be284776f41900661182', 'title': 'University_of_Notre_Dame', 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}

    ```

* Another Method is to load using ***torchtext***
    ```python
    ## torch code
    from torchtext.datasets import DBpedia as db

    train_iter = db(split='train')
    ```


### YAGO

YAGO is a knowledge base, i.e., a database with knowledge about the real world. YAGO contains both entities (such as movies, people, cities, countries, etc.) and relations between these entities (who played in which movie, which city is located in which country, etc.). All in all, YAGO contains more than 50 million entities and 2 billion facts.

<center class="half">
<img src=./Pictures/Build_a_knowledge_graph/figure1.png width = 70%>
</center>


* YAGO is stored in the standard Resource Description Framework “RDF”. This means that YAGO is a set of facts, each of which consists of a subject, a predicate (also called “relation” or “property”) and an object — as in (Elvis) (birthPlace) (Tupelo).



### OpenCyc

* The Cyc knowledge base from Cycorp contains about 1.5 million general concepts and more than 20 million general rules, with an accessible version called OpenCyc deprecated sine 2017.

* the language of Cyc is **CycL**.


### Freebase
* On 16 December 2015, Google officially announced the Knowledge Graph API, which is meant to be a replacement to the Freebase API. Freebase.com was officially shut down on 2 May 2016.


### NELL

* NELL is built from the Web via an intelligent agent called Never-Ending Language Learner. It has 2,810,379 beliefs with high confidence by far.

* Since January 2010, our computer system called NELL (Never-Ending Language Learner) has been running continuously, attempting to perform two tasks each day:

    * First, it attempts to "read," or extract facts from text found in hundreds of millions of web pages (e.g., playsInstrument(George_Harrison, guitar)).

    * Second, it attempts to improve its reading competence, so that tomorrow it can extract more facts from the web, more accurately.


### Wikidata
* Wikidata is a free structured knowledge base, which is created and maintained by human editors to facilitate the management of Wikipedia data. It is multi-lingual with 358 different languages.



### Google KG
Python code to visit the Google knowledge graph
```python
## first, need to config the google API
can visit the link here: https://developers.google.com/knowledge-graph/reference/rest/v1 

"""Example of Python client calling Knowledge Graph Search API."""
import json
import urllib

api_key = open('.api_key').read()
query = 'Taylor Swift'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
params = {
    'query': query,
    'limit': 10,
    'indent': True,
    'key': api_key,
}
url = service_url + '?' + urllib.urlencode(params)
response = json.loads(urllib.urlopen(url).read())
for element in response['itemListElement']:
  print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
```


### Probase
* Microsoft builds a probabilistic taxonomy called Probase with 2.7 million concepts. 
* the official doc has rich description of the advantages of their datasets




## Domain-specific datasets

|||||||



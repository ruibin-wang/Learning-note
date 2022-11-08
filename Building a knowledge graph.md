# Building a knowledge graph

the following description comes from the tutorial link: https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/#h2_6  and code: https://colab.research.google.com/drive/1YTv-9ENIeVWCGqvjwVSTfChAaj05vV0h#scrollTo=RtN8y8chs3zJ

## Step 1: Entities Extraction

*  use the parts of speech (POS) tags. The Nouns and the proper nouns would be entities.

* This kind of method  called 'dependency parsing'.

* code can be found as follows:
```python
## First, install packages
pip install spacy
python -m spacy download en ## very necessary, needed when loading the data.

## test code
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp("The 22-year-old recently won ATP Challenger tournament.")

for tok in doc:
    print(type(tok))
    print(tok.text, "...", tok.dep_)


## output
The ... det
22 ... nummod
- ... punct
year ... npadvmod
- ... punct
old ... nsubj
recently ... advmod
won ... ROOT
ATP ... compound
Challenger ... compound
tournament ... dobj
. ... punct
```


## Step2: Extract Relations

* the relation of two entities can also be extracted through 'dependency parsing' by using the Root of the sentences (which is also the verb of the sentence). **Rule-based method!**

    * code for this rule-based method can be found as follows:
        ```python
        ## code
        doc = nlp("Nagal won the first set.")

        for tok in doc:
            print(tok.text, "...", tok.dep_)


        ## output
        Nagal ... nsubj
        won ... ROOT
        the ... det
        first ... amod
        set ... dobj
        . ... punct

        ```


## Build a graph

Two method to build this graph,

* one is to use the .csv data, and networkx package

    ```python
    import networkx as nx
    # extract subject
    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]

    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

    # create a directed-graph from a dataframe
    G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                            edge_attr=True, create_using=nx.MultiDiGraph())


    plt.figure(figsize=(12,12))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    plt.show()
    ```

* another method is to use the .csv data to build the graph in Neo4j database. code can be found in my [github](https://github.com/ruibin-wang/web_crawler_graph_building/blob/main/json2neo.py)

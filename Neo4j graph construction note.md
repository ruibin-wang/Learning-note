# Neo4j graph construction note

Before using the two packages, download the Neo4j desktop file and install it. Then build a new project, set a **password** and open it in its own browser, find the local host address and username.



## Neo4j package

### code and turtorial

* The code can be find in the file "neo4j_data_construct.py" at: https://github.com/ruibin-wang/web_crawler_graph_building.git

* The turtorial can be found at: https://www.youtube.com/watch?v=IShRYPsmiR8&t=527s 

* doc can be found at: https://neo4j.com/developer/get-started/


### connection

```python
from neo4j import GraphDatabase
graph = GraphDatabase.driver(uri="bolt://localhost:7687", auth=("username", "password"))

session = graph.session()
session.run('Neo4j command')

```


### create nodes

```neo4j
CREATE (n)  ## create a node without attribute
CREATE (n:LABEL{attibute1: 'value1', attibute2: 'value2',}) 


Example:

CREATE (n:Person{name: 'honey', favoratecolor:'red'})

```


### match the node


```neo4j
## dispplay the matched items
MATCH (n) RETURN (n) LIMIT + number  


## select the entities with the name "LABEL"
MATCH (n:LABEL) RETURN (n)
MATCH (n:LABEL) RETURN (n) LIMIT + number 


Example:

MATCH (n) Return (n) LIMIT 10 

```


### delete the node

```
## delete the relationship and the node in the dataset
MATCH (n) DETACH DELETE n 

## delete the nodes

MATCH (n) DELETE (n)

## conditional match
MATCH (s:LABEL1), (p:LABEL2) 
WHERE s.attribute1 = 'value1' and p.attribute2 = 'value2'  
RETURN s,p

```


### relationship 

```
CREATE (node1)-[R:name_of_relationship]->(node2) 

CREATE (node1)-[R:name_of_relationship{attri1: attri_val1}]->(node2) 

Example:


MATCH (s:LABEL1), (p:LABEL2) 
WHERE s.attribute1 = 'value1' and p.attribute2 = 'value2'
CREATE (p)-[R:studeied_at]->(s)
```

## py2neo package


### code and turtorial

* code can be find in the file "py2neo_package_graph.py" at: https://github.com/ruibin-wang/web_crawler_graph_building.git

* tourtial can be found at: https://cloud.tencent.com/developer/article/1434904 and https://cuiqingcai.com/4778.html 


## connection

```python
graph = Graph("bolt://localhost:7687", auth=("username", "password"))
```


## Node & Relationship

```python
from py2neo import Node, Relationship, Graph 

node1 = Node(label1, attr1 = value1)

r = Relationship(node1, relation_name, node2)


Example

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)


## node and relationship are similar to the dictionary, it can be edited by following

a[key] = value
a.setdefault(key, value)
a.update(dict)

```


## subgraph

```python

s = a | b | c

## we can use the following command to get the Key、Label、Node、Relationship、Relationship Type of the subgraph
s.keys()
s.labels()
s.nodes()
s.relationships()
s.types() 

# add the subgraph object to the existing dataset
graph.create(r)  

```

## walkable

```python

# use the function 'walk'

from py2neo import walk

w = Relationship(a,'re',b) + Relationship(a,'re',c) + Relationship(c,'re',b) 

## walkable object
walk_var = walk(w)


# use the following code to get Node、start Node、all Node and Relationship 
w.start_node ()
w.end_node ()
w.nodes ()
w.relationships ()
```


## Query

```python
## three function can select the nodes and the relationship

graph.run()
graph.data()

## use data() to run the CQL 

data = graph.data('MATCH (n) RETURN n')
data = graph.run('MATCH (n) RETURN n')


## use find() and find_one() to select nodes
node = graph.find_one(label="LABEL")


## use match and match_one() to select relationships
relationship = graph.match_one(rel_type = "relation")

## use NodeMatcher() to select the nodes
from py2neo import Graph, NodeMatcher
selector = NodeMatcher(graph)

nodes = selector.select('LABEL', attribute=value)

# use order_by to order the seq

nodes = selector.select('LABEL', attribute=value).order_by('_.attribute')




```











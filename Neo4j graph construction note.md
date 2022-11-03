# Neo4j graph construction note

## code and turtorial

* The code can be find in the file "neo4j_data_construct.py" at: https://github.com/ruibin-wang/web_crawler_graph_building.git

* The turtorial can be found at: https://www.youtube.com/watch?v=IShRYPsmiR8&t=527s 



## create nodes

```neo4j
CREATE (n)  ## create a node without attribute
CREATE (n:LABEL{attibute1: 'value1', attibute2: 'value2',}) 


Example:

CREATE (n:Person{name: 'honey', favoratecolor:'red'})

```




## match the node


```neo4j
## dispplay the matched items
MATCH (n) RETURN (n) LIMIT + number  


## select the entities with the name "LABEL"
MATCH (n:LABEL) RETURN (n)
MATCH (n:LABEL) RETURN (n) LIMIT + number 


Example:

MATCH (n) Return (n) LIMIT 10 

```


## delete the node

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


## relationship 

```
CREATE (node1)-[R:name_of_relationship]->(node2) 

CREATE (node1)-[R:name_of_relationship{attri1: attri_val1}]->(node2) 

Example:


MATCH (s:LABEL1), (p:LABEL2) 
WHERE s.attribute1 = 'value1' and p.attribute2 = 'value2'
CREATE (p)-[R:studeied_at]->(s)



```







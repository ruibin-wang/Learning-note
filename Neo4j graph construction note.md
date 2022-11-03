# Neo4j graph construction note


## create nodes

```neo4j
CREATE (n) RETURN n  ## create a node without attribute
CREATE (n:LABEL{attibute1: 'value1', attibute2: 'value2',}) RETURN n


Example:

CREATE (n:Person{name: 'honey', favoratecolor:'red'})

```




## match the node


```neo4j
## dispplay the matched items
MATCH (n) RETURN n LIMIT + number  


Example:

MATCH (n) Return n LIMIT 10 

```


## delete the node

```
## delete the relationship and the node in the dataset
MATCH (n) DETACH DELETE n 

## delete the nodes

MATCH (n) DELETE n


```


<font size=3> **neo4j** </font>






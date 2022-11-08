# Building a knowledge graph

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

* the relation of two entities can also be extracted through 'dependency parsing' by using the Root of the sentences (which is also the verb of the sentence).

* code can be found as follows:
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





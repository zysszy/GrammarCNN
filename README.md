# GrammarCNN
## Usage
### To train new model
In folder ```model/```, train new model
```
python3 run.py train [train|dev|test] [tree|var|func]
```
```tree``` for nonterminal nodes, ```var``` for variable nodes, and ```func``` for function nodes.

```train```, ```dev``` and ```test``` denote the evaluation set.
### To predict
After ```tree```, ```var```, and ```func``` are trained.

In folder ```predict/```
```
python3 run.py [pre|eval]
```
## Dependenices 
  * NLTK 3.2.1
  * Tensorflow 1.3.1
  * Python 3.5
  * Ubuntu 16.04

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
  * Java 1.8
  
# Examples
We successly generated.
 ```
class BootyBayBodyguard(MinionCard ) : 
    def __init__ (self) :
        super().__init__("Booty Bay Bodyguard", 5, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON)
    def create_minion (self, player) :
        return Minion(5, 4, taunt = True)
```
Example Code:
```
class BootyBayBodyguard(MinionCard ) : 
    def __init__ (self) :
        super().__init__("Booty Bay Bodyguard", 5, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON)
    def create_minion (self, player) :
        return Minion(5, 4, taunt = True)
```
 Our model tends to generate a structrual correct code, which leads to a higher StrAcc but a similar BLEU compared with previous works.
 
 Code we generated.
```
class AnnoyoTron(MinionCard ) : 
    def __init__ (self) :
        super().__init__("Annoy-o-Tron", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, minion_type = MINION_TYPE.MECH, divine_shield = True)
    def create_minion (self, player) :
        return Minion(1, 2, taunt = True, divine_shield = True)
```
Example Code:
```
class AnnoyoTron(MinionCard ) : 
    def __init__ (self) :
        super().__init__("Annoy-o-Tron", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, minion_type = MINION_TYPE.MECH)
    def create_minion (self, player) :
        return Minion(1, 2, divine_shield = True, taunt = True)
 ```

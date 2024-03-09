# Accuracy Results of HW4 - Multi-Classs Image Classification using CNN


**dataset:**

|                  |unique labels  |total images      |
|------------------|---------------|------------------|
| train set        |34             |3056              |
| test set         |34             |2149              |

* ***The number of unique labels is the number of classes!***


### * Not pretrained CNN:
* training epochs: 30
* batch size: 40
* optimizer: Adam
* initial lr: 0.0001
* loss: categorical crossentropy
* data augmentations: {rotation, horizontal/vertical flip, brightness, zoom}

|                  |loss           |accuracy          |
|------------------|---------------|------------------|
| train set        |0.3277         |0.9064            |
| validation set   |0.3536         |0.9098            |
| test set         |1.5507         |0.7073            |


* ***The validation set occured from a split at the initial train set with a validation split percentage 0.2!***


### * Pretrained MobileBet:
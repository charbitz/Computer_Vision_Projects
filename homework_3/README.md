# Accuracy Results of HW3 - Multi-Classs Image Classification using BOVW

**classes:**
|  motorbike  | school-bus  | touring-bike  | airplane  | car  |
|---------|---------|---------|---------|---------|
| ![](https://github.com/charbitz/Computer_Vision_Projects/blob/master/homework_3/caltech/imagedb/145.motorbikes-101/145_0013.jpg)     |     ![](https://github.com/charbitz/Computer_Vision_Projects/blob/master/homework_3/caltech/imagedb/178.school-bus/178_0018.jpg)    |    ![](https://github.com/charbitz/Computer_Vision_Projects/blob/master/homework_3/caltech/imagedb/224.touring-bike/224_0022.jpg)     |  ![](https://github.com/charbitz/Computer_Vision_Projects/blob/master/homework_3/caltech/imagedb/251.airplanes-101/251_0026.jpg)       |   ![](https://github.com/charbitz/Computer_Vision_Projects/blob/master/homework_3/caltech/imagedb/252.car-side-101/252_0030.jpg)      |

### * k-NN Classifier:

* for 50 optical words:

|               |  k = 5  | k = 20  | k = 40  | k = 60  | k = 80  | k = 100 |
|---------------|---------|---------|---------|---------|---------|---------|
| motorbike     |<0.5>      |0.1      |0.1      |0.3      |0.2      |0.1      |
| school-bus    |**0.777**    |0.444    |0.444    |0.666    |0.666    |**0.777**    |
| touring-bike  |**1**        |0.818    |0.818    |0.818    |0.818    |0.636    |
| airplane      |0.818    |**1**        |**1**        |**1**        |**1**       |**1**       |
| car           |0.454    |0.545    |0.545    |0.545    |0.636    |**0.727**    |
| global        |**0.711**    |0.596    |0.596    |0.673    |0.673    |0.653    | 




### * SVM Classifier:
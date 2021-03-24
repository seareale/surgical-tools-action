# surgical-tools-action
based on ResNet, add 2 FC layers in the end of model.  
use **ImbalancedDatasetSampler** when the data is loading.  
- input : 224X224  
- l-rate : 0.01  
- batch-size : 128
- weight decay : 5e-4 

## Install
### Requirements
Using ```Imbalanced Dataset Sampler``` from [ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)
```bash
$ git clone https://github.com/ufoym/imbalanced-dataset-sampler.git
$ cd imbalanced-dataset-sampler
$ python setup.py install
$ pip install .
$ cd ..
```
```bash
$ git clone https://github.com/SeaRealE/surgical-tools-action.git
$ cd surgical-tools-action
$ pip install -r requirements.txt  
``` 
### Preparing dataset
- you can prepare datasets in [here](dataset/README.md)  

---
## Training
```bash
$ python3 run.py --gpu ${GPU_NUM} --epoch ${EPOCH} --lrate ${LEARNING_RATE} \
                        --size ${INPUT_SIZE} --batch ${BATCH_SIZE}
```  
ex)   
```$ python3 run.py```  
```$ python3 run.py --gpu 6 --epoch 30 --lrate 0.01 --size 224 --batch 128```

---
## Result
#### Distribution of surgical tool image
&nbsp; | close | open |
---- | ---- | ----
train | 57,295 | 14,640
val | 6,838 | 1,926
test | 7,112 | 2,226

#### Accuracy for surgical tool action recognition
Model | # of params | Acc |
---- | ---- | ----
ResNet18 | 11M |0.8287
ResNet34 | 21M | 0.8360
ResNet50 | 25M | 0.8658
ResNet101 | 44M | 0.8588
ResNet152 | 60M | 0.8609
**ResNeXt50_32x4d** | **25M** | **0.8696**
ResNeXt101_32x8d | 88M | 0.8555
Wide_ResNet50_2 | 68M | 0.8503
Wide_ResNet101_2 | 126M | 0.8549

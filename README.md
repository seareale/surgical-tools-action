# surgical-tools-action-classification
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

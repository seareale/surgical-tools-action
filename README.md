# surgical-tools-action
## Install
### Requirements
```bash
$ pip install -r requirements.txt  
$ git clone https://github.com/SeaRealE/surgical-tools-action.git
``` 
### Preparing dataset
- you can prepare datasets in [here](dataset/README.md)  

---
## Training
```bash
$ python3 res101_run.py ${GPU_NUM} ${LABEL_IDX} --epoch ${EPOCH} --lrate ${LEARNING_RATE} \
                        --size ${INPUT_SIZE} --batch ${BATCH_SIZE}
```  
ex)   
```$ python3 res101_run.py 6 11```  
```$ python3 res101_run.py 6 11 --epoch 30 --lrate 0.01 --size 224 --batch 128```

## Inference
```bash
$ python3 test.py ${LABEL_IDX} ${CHECKPOINT_PATH} --size ${INPUT_SIZE} --batch ${BATCH_SIZE}
```  
ex)   
```$ python3 res101_run.py 11 ./save/[CHECKPOINT].pth```  
```$ python3 res101_run.py 11 ./save/[CHECKPOINT].pth --size 224 --batch 128```

---
## Result

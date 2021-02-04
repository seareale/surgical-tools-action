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
$ python3 run.py --gpu ${GPU_NUM} --epoch ${EPOCH} --lrate ${LEARNING_RATE} \
                        --size ${INPUT_SIZE} --batch ${BATCH_SIZE}
```  
ex)   
```$ python3 run.py```  
```$ python3 run.py --gpu 6 --epoch 30 --lrate 0.01 --size 224 --batch 128```

---
## Result

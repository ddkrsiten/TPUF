# TPUF
This is the repository of our work 《TPUF: Enhancing Cross-domain Sequential Recommendation via Transferring Pre-trained User Features》
## Requirements
* Python 3.7.16
* pandas==1.3.5
* scipy==1.7.3
* torch==1.13.0+cu117

You may use "pip install -r requirements.txt" to install all the required libraries.
## Usage

### Datasets
The datasets in the repository are the preprocessed versions that follow the rules mentioned in the paper.

The original datasets could be downloaded from the following website：

http://jmcauley.ucsd.edu/data/amazon/index_2014.html

https://nijianmo.github.io/amazon/index.html

https://github.com/FengZhu-Joey/GA-DTCDR

Note: Regarding Amazon Movies-Books datasets, we used the 2014 version, while regarding Amazon Sports-Clothing datasets, we used the 2018 version.

### Preparation
The models in "s_model" file are the pre-trained source models including MF, MLP and SASRec. As there are numerous implementations of MF and MLP available online, we do not provide training process for these two types of models here. However, you could still use them as source-domain models to train TPUF.

If you want to retrain SASRec as the source-domain model, you can use the following code:
```
nohup python -u main.py --dataset douban_movie --maxlen 50 --model SASRec --l2_emb 0.0 --hidden_units 32 --num_epochs 200 --gpu 0 > logs/a.logs 2>&1 &
```
```
nohup python -u main.py --dataset Amazon_movie --maxlen 50 --model SASRec --l2_emb 0.0 --hidden_units 32 --num_epochs 200 --gpu 0 > logs/b.logs 2>&1 &
```
```
nohup python -u main.py --dataset Amazon_sport --maxlen 50 --model SASRec --l2_emb 0.0 --hidden_units 32 --num_epochs 200 --gpu 0 > logs/c.logs 2>&1 &
```

### Training
Please set --model TPUF while trainning TPUF model.

All the recommended parameters are recorded in "run.sh". You could run the commands in "run.sh" to reproduce our experiments.

# Offline_Evaluation_in_RecommenderSystems

- *What is the impact of data leakage in recommendation? Are we expecting better recommendation accuracy by using more “future data” in training?*
- *To what extent can we compare model performance if data leakage exists in offline evaluation?*

___

Mainly we focused on the impact of data leakage on Recommender Systems accuracy.

We reproduced the results using the following implementations that were implemented by:

**Data Leakage in Recommender System Folder** (Main)
1. Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li (2020). [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060).[github](https://github.com/dataLeakageRec/dataLeakageRec)

**Neural Collaborative Filtering Folder**

2. [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060) adopted the implementation from the authors Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017) Neural Collaborative Filtering [Paper](https://arxiv.org/abs/1708.05031), [github](https://github.com/hexiangnan/) and verified that the code is able to reproduce the results in the original paper if following their experimental settings.


## **Models**
- Bayesian Personalized Ranking(BPR) 
- Neural Matrix Factorization (NueMF)


## **Dataset used**
- Movielens-25M
- Amazon-music

## Data format

| User ID | Item ID | Rating (1 or 0) | Timestamp | Year | 
| ------- |:-------:|:---------------:| ----------|----- |
| ------- | ------- | --------------- | ----------|----- |
| ------- | ------- | --------------- | ----------|----- |

***

##  Data Leakage folder
### **Environment setting for the Data Leakage folder**
- Tensorflow 1.14
- Python 3.6.9
- Keras 2.3.0

### **Example to run the codes in the Data Leakage Folder**

- For Running the **NeuMF** model:

```python
python NeuMF_test.py --path [path of the data folder]  --data movielens --selected_year 4 --num_years_added 0 --gpu 1 --regs 0.01 --learning_rate 0.001 --num_negatives 4 --mf_dim 64
```

- For Running the **BPR** model:

```python
python test.py --path [path of the data folder] --data movielens --selected_year 4 --num_years_added 0 --gpu 1 --factors 64 --learning_rate 0.001 --reg 0.01
```
***

## Neural Collaborative Filtering Folder

### **Environment setting for the Neural Collaborative Filtering folder**
They use Keras with Theano as the backend.

- Keras version: '1.0.7'
- Theano version: '0.8.0'



### **Example to run the codes in the Neural Collaborative Filtering Folder**

- For Running the NeuMF model((without pre-training)):

```python
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

- For Running the BPR model(with pre-training):

```python
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5

```
***

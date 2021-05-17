# Offline_Evaluation_in_RecommenderSystems

- *What is the impact of data leakage in recommendation? Are we expecting better recommendation accuracy by using more “future data” in training?*
- *To what extent can we compare model performance if data leakage exists in offline evaluation?*

Mainly we focused on the impact of data leakage on Recommender Systems accuracy.


We reproduced the results that were implemented by:

Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li (2020). [A Critical Study on Data Leakage in Recommender System Offline Evaluation](https://arxiv.org/abs/2010.11060).

They adopted the implementation from the authors Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017) Neural Collaborative Filtering [Paper](https://arxiv.org/abs/1708.05031), [github](https://github.com/hexiangnan/) and verified that the code is able to reproduce the results in the original paper if following their experimental settings.



# **Models**
- Bayesian Personalized Ranking(BPR) 
- Neural Matrix Factorization (NueMF)


# **Dataset used**
- Movielens-25M
- Amazon-music

## Data format

| User ID | Item ID | Rating (1 or 0) | Timestamp | Year | 
| ------- |:-------:|:---------------:| ----------|----- |
| ------- | ------- | --------------- | ----------|----- |
| ------- | ------- | --------------- | ----------|----- |


## **Environment setting**
- Tensorflow 1.14
- Python 3.6.9
- Keras 2.3.0

***


### **Example to run the codes**

- For Running the NeuMF model:

```python
python NeuMF_test.py --path [path of the data folder]  --data movielens --selected_year 4 --num_years_added 0 --gpu 1 --regs 0.01 --learning_rate 0.001 --num_negatives 4 --mf_dim 64
```

- For Running the BPR model:

```python
python test.py --path [path of the data folder] --data movielens --selected_year 4 --num_years_added 0 --gpu 1 --factors 64 --learning_rate 0.001 --reg 0.01
```
***

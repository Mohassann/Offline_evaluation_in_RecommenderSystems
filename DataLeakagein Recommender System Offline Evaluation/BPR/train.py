
from dataloader import LoadRatingFile_HoldKOut_year
import numpy as np
import argparse

import time
import tensorflow as tf

import heapq
import random



parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
parser.add_argument('--selected_year', type=int, default = 5, help='selected_year')
parser.add_argument('--num_years_added', type=int, default= 0, help='num_years_added')
parser.add_argument('--data', type=str, default = 'ml')
parser.add_argument('--gpu', default= '4', help='gpu')
parser.add_argument('--factors', type=int, default = 64, help = 'embedding dimensions')
parser.add_argument('--learning_rate', type=float, default = 0.0001, help = 'learning rate')
parser.add_argument('--reg', type=float, default=0.00001, help='regularization coefficient')


args = parser.parse_args()


selected_year = int(args.selected_year)
num_years_added = int(args.num_years_added)
data = args.data


splitter = '\t'
hold_k_out = 1

data_path = args.path


train, test, available_items, num_user, num_item, num_ratings = LoadRatingFile_HoldKOut_year(data_path+data+'/', splitter, hold_k_out,selected_year, num_years_added)


import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def bpr_mf(user_count, item_count, hidden_dim, learning_rate, regularization_rate):
    
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])
    i_test = tf.placeholder(tf.int32, [None])
    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.01))
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.01))
    
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    
    i_test_emb = tf.nn.embedding_lookup(item_emb_w, i_test)
    output_rating = tf.matmul(u_emb, tf.transpose(i_test_emb))
    
    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)

    mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])


    bprloss = regularization_rate * l2_norm - tf.reduce_sum(tf.log(tf.sigmoid(x)))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(bprloss)

    return u, i, j, bprloss, train_op, output_rating, i_test



import random
def get_batch(train, available_items, num_items):
    users, pos_items, neg_items = [], [], []
    for user in train.keys():
        for item in train[user]:
            users.append(user)
            pos_items.append(item)
            j = random.sample(available_items-set(train[user]),1)[0]
            neg_items.append(j)
            
    return (users, pos_items, neg_items)






def getNDCG(k, preds, y):
    ndcg = 0
    for j in range(0, k):
        if preds[j] == y:
            ndcg = 1 / np.log2(j + 2)
    return ndcg



factors = args.factors
learning_rate = args.learning_rate
reg = args.reg

num_epochs = 100
batch_size = 1024

verbose = 20

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

with tf.Session(config=config) as sess:
    tf.debugging.set_log_device_placement(True)
    u, i, j, bprloss, train_op, output_rating, i_test = bpr_mf(num_user, num_item, factors, learning_rate, reg)
    
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        losses = []
        u_,i_,j_ = get_batch(train ,available_items,num_item)
        for s in range(np.int(num_ratings / batch_size)+1):
            loss,_ = sess.run([bprloss, train_op], feed_dict={u: u_[s*batch_size:(s+1)*batch_size], 
                                                          i: i_[s*batch_size:(s+1)*batch_size], 
                                                          j: j_[s*batch_size:(s+1)*batch_size]})
            losses.append(loss)
        print(epoch, np.mean(losses))

        if epoch%verbose == 0:
            HR10 = []
            NDCG10 = []
            k = 10

            for test_i in range(0, len(test)):
                user = test[test_i][0]
                target = test[test_i][1]
                negative = list(available_items)
                output = sess.run(output_rating, feed_dict = {u:[user], i_test: negative})[0]
                preds = dict(zip(negative, output))
                recommended = heapq.nlargest(k, preds, key = preds.get)
                if target in recommended:
                    HR10.append(1)
                else:
                    HR10.append(0)

                NDCG10.append(getNDCG(k, recommended, target)) 

            hr = np.mean(HR10)
            ndcg = np.mean(NDCG10)
            print(epoch,hr,ndcg)

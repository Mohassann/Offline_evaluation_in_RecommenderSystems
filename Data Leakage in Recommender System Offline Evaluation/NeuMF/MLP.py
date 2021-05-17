import numpy as np
import warnings
warnings.simplefilter("ignore", category = Warning)



import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, multiply, Reshape, concatenate, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import argparse
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--num_years_added', nargs='?', type=int, default= 0,
                        help='Number of years added.')
    parser.add_argument('--selected_year', nargs = '?', type =int, default = 5, help='selected year')
    parser.add_argument('--gpu', nargs='?', default = '5', help='gpu')
    parser.add_argument('--data', type=str, default = 'movielens', help='dataset')
    return parser.parse_args()


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        for t in np.arange(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def init_normal(shape, name=None):
    return initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0)

def get_model(num_users, num_items, layers = [20,10], reg = 0):
    
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                  embeddings_initializer = init_normal([num_users, int(layers[0]/2)]), embeddings_regularizer = l2(reg), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                  embeddings_initializer = init_normal([num_users, int(layers[0]/2)]), embeddings_regularizer = l2(reg), input_length=1)
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = concatenate([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model


if __name__ == '__main__':
    args = parse_args()
    selected_year = args.selected_year
    data = args.data
    num_years_added = args.num_years_added
    dataset = Dataset(path, selected_year, num_years_added)
    train, testRatings, available_items, trainList = dataset.trainMatrix, dataset.testRatings, dataset.available_items, dataset.trainList
    num_users, num_items = train.shape

    import math
    def getHitRatio(ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    learner = 'adam'
    layers = [64,32,16,8]
    reg = 0.000001
    num_negatives = 4
    learning_rate = 0.01


    batch_size = 1024 
    epochs = 100
    verbose = 1


    model = get_model(num_users, num_items, layers, reg)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    


    for epoch in np.arange(epochs):
        user_input, item_input, labels = get_train_instances(train, num_negatives)
    hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        if epoch %verbose == 0:
            hits = []
            ndcgs = []
            time_0 = time()
            K = 10
            for idx in range(len(testRatings)):
                user = testRatings[idx][0]
                gtItem = testRatings[idx][1]
                items = list(set(available_items) - set(trainList[user]))#_testNegatives[idx]
                users = np.full(len(items), user, dtype = 'int32')
                
                predictions = model.predict([users, np.array(items)], 
                                     batch_size=1024, verbose=0)
                predictions = np.squeeze(predictions)
                map_item_score = dict(zip(items, predictions))
                ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
                hr = getHitRatio(ranklist, gtItem)
                ndcg = getNDCG(ranklist, gtItem)
                hits.append(hr)
                ndcgs.append(ndcg)
            hr = np.mean(hits)
            ndcg = np.mean(ndcgs)

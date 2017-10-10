#################
# Task 11
#################
import numpy as np
import random

NUM_ITERATIONS = 25
def kmeans_vec(data, K):
    centroids = init_centroids(K, data)
    for i in range(NUM_ITERATIONS):
        index = assign_step(data, centroids)

        centroids = update_step(data, index, K)
    return centroids


def init_centroids(K, data):
    data = np.array(data)
    index = random.sample(range(len(data)), K)
    centroids = data[index,:]
    return centroids

def assign_step(data, centroids):
    data = np.array(data)
    centroids = np.array(centroids)

    centroids = np.expand_dims(centroids,axis=1)
    distance = np.sqrt((data - centroids)**2)
    sum_distance = distance.sum(axis = 2)

    index = np.argmin(sum_distance,axis = 0)
    return index

def update_step(data, index, K):
    dimension = len(data[0])
    sumvalues = np.zeros((K,dimension))
    centroids = np.zeros((K,dimension))
    count = np.zeros((K,dimension))

    for i in range(len(data)):
        count[index[i]] += 1
        sumvalues[index[i]] += data[i]
    centroids = sumvalues/count
    return centroids



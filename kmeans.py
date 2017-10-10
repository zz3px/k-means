from math import sqrt
import numpy as np
import random

NUM_ITERATIONS = 25
def kmeans(data, K):
    centroids = init_centroids(K, data)
    for i in range(NUM_ITERATIONS):
        index = assign_step(data, centroids)
        centroids = update_step(data, index, K)
    return centroids

def distance(a, b):
    dist = 0
    for i, j in zip(a, b):
        dist += (i - j)**2
    return sqrt(dist)

def init_centroids(K, data):
    data = np.array(data)
    index = random.sample(range(len(data)), K)
    centroids = data[index,:]
    return centroids

def assign_step(data, centroids):

    n = len(data)
    K = len(centroids)

    index = [0] * n
    for i in range(n):
        update = distance(data[i], centroids[0])
        for j in range(1, K):
            dis = distance(data[i], centroids[j])
            if dis < update:
                index[i] = j
                update = dis
    return index

def update_step(data,index,K):
    dimension = len(data[0])
    sumofpoints = [[0] * dimension for x in range(K)]
    centroids = [[0] * dimension for x in range(K)]
    count = K * [0]
    for i in range(len(data)):
        count[index[i]] += 1
        for j in range(dimension):
            sumofpoints[index[i]][j] = data[i][j] + sumofpoints[index[i]][j]

    for i in range(len(data)):
        for j in range(dimension):
            centroids[index[i]][j] = sumofpoints[index[i]][j] / count[index[i]]

    return centroids


import os, sys, matplotlib.pyplot as plt, numpy as np, itertools, math
from random import seed
from random import random, randint
from scipy.spatial import distance
from  tqdm import tqdm
import random
import warnings
from math import factorial





'''
Generate positive pairs. In a positive pair, both samples belong to the same class. 
Parameters:
    
    INPUT
                   X : Input features, Type: nd array
                   y : class labels, Type: 1d numpy array
             rand_samples : Number of random samples taken from each class, Type: integer. By default it selects all samples
            pair_len : Number of random pairs from each class, Type: integer. By default it selects all pairs

    OUTPUT
            anchor and pos are pairs of features.

'''
def generate_positive_pairs(X, y, rand_samples, pair_len) :

    # Shape of features
    row, col = X.shape[0], X.shape[1]

    # Get unique elements of y array along with their frequencies
    uniq, freq = np.unique(y, return_counts=True)

    anchor = []
    pos = []
    count = 0

    # Traverse through each class and select random samples
    for x, f in zip(uniq, freq) :

        # Select indices of a certian class label
        ind, = np.where(y == x)
        ind = list(ind)

        # Select random samples indices. If number of samples required are more than frequency of class, then select all samples
        if rand_samples <= f and rand_samples != -1:
            random_indices = random.sample(ind, rand_samples)
        elif rand_samples == -1 :
            random_indices = ind
        else :
            warnings.warn("ValueError ! 'samples' required are more than number of elements in class. So, all elements are selected.")
            random_indices = random.sample(ind, f)

        # Generate positive pairs
        pairs = list(itertools.combinations(random_indices, 2))

        if len(pairs) >= pair_len :
            pairs = list(pairs)[:pair_len]

        # print(len(pairs))

        # Get features associated with those indices
        anchor += [X[i] for i, _ in pairs]
        pos += [X[i] for _, i in pairs]    

    # Convert features to numpy matrix
    anchor = np.array(anchor)
    a = (len(anchor),)
    b = tuple(X[0].shape)
    anchor = anchor.reshape(a+b)

    a = (len(anchor),)

    pos = np.array(pos)
    pos = pos.reshape(a+b)

    # print(positive_extra[0].shape)

    return anchor, pos, uniq, freq






'''
Generate negative pairs. In a negative pair, both samples belong to the same class. 
Parameters:
    
    INPUT
                   X : Input features, Type: nd array
                   y : class labels, Type: 1d numpy array
             rand_samples : Number of random samples taken from each class, Type: integer. By default it selects all samples
            pair_len : Number of random pairs from each class, Type: integer. By default it selects all pairs

    OUTPUT
            neg and neg_X2 are pairs of features.

'''

def generate_negative_pairs(X, y, uniq, freq, rand_samples, pair_len) :

    # Shape of features
    row, col = X.shape[0], X.shape[1]

    # Get unique elements of y array along with their frequencies
    # uniq, _ = np.unique(y, return_counts=True)
    indices = []

    # Traverse through each class and select random samples
    for x, f in zip(uniq, freq) :

        # Select indices of a certian class label
        ind, = np.where(y == x)
        ind = list(ind)

        # Select random samples indices. If number of samples required are more than frequency of class, then select all samples
        if rand_samples <= f and rand_samples != -1:
            random_indices = random.sample(ind, rand_samples)
        elif rand_samples == -1 :
            random_indices = ind
        else :
            warnings.warn("ValueError ! 'samples' required are more than number of elements in class. So, all elements are selected.")
            random_indices = random.sample(ind, f)

        # Get features associated with those indices
        indices.append(random_indices)   
        # print(len(random_indices))

    neg = []

    # Generate negative pairs
    for i in range(len(uniq)) :

        # Generate pair of a class with every other class
        for j in range(len(uniq)) : 

            if i == j :
                continue
            

            curr = indices[j]
            num = math.ceil(pair_len/(len(uniq)-1))
            while len(curr) < num :
                curr += indices[j]

            indices1 = random.sample(curr, num)
            indices1 = indices1[:pair_len]
            # Get features associated with those indices
            neg += [X[p] for p in indices1]


    # Convert features to numpy matrix
    neg = np.array(neg)
    a = (len(neg),)
    b = tuple(X[0].shape)

    neg = neg.reshape(a+b)


    return neg






# Computed number of combinations nCr
def calculate_combinations(n, r):
    return factorial(n) // factorial(r) // factorial(n-r)






'''
Generate positive and negative pairs. This function selects samples in such a way that nmber fo apositive pairs equals number 
of negative pairs. Only exception is when the numbers of samples are less than total combinations of all classes.
Parameters:
    
    INPUT
                   X : Input features, Type: nd array
                   y : class labels, Type: 1d numpy array
             rand_samples : Number of random samples taken from each class, Type: integer. By default it selects all samples
            pos_pair_size : Number of positive random pairs from each class, Type: integer. By default it selects all pairs

    OUTPUT
            anchor and pos are positive pairs of features.
            neg and neg_X2 are negative pairs of features.

'''

def generate_pairs(X, y, rand_samples, pos_pair_size=-1, extra_data=[]) :


    # print("Input ", len(extra_data))
    uniq, f = np.unique(y, return_counts= True)

    N = len(uniq)
    pair_neg = calculate_combinations(N, 2)
    pair_pos = calculate_combinations(min(f), 2)

    pos_pair_size = min(pos_pair_size, pair_pos )

    if pos_pair_size == -1 :
        if rand_samples == -1 :
            # pos_pair_size = calculate_combinations(int(len(y)/N), 2)
            pos_pair_size = calculate_combinations(int(min(f)), 2)
        else :
            # pos_pair_size = min(calculate_combinations(rand_samples, 2), int(len(y)/N) )
            pos_pair_size = min(calculate_combinations(rand_samples, 2), pair_pos )

    pos_pair_size = int(pos_pair_size)
    # print("pos pair size ", pos_pair_size)
    for i in range(pos_pair_size, 0, -1):
        if ((N/ pair_neg) * i).is_integer() :
            break

    neg_samples = int((N/ pair_neg) * i)
    pos_samples = i

    if ((N/ pair_neg) * i).is_integer() is False :
        warnings.warn("Number of samples per class are less than total combinations of all classes. 1 sample will be selected from each negative pair. ")
        neg_samples = 1

    # print(N, pair_neg, neg_samples, i)
    # print(pair_neg, neg_samples, pos_pair_size, i)
    # print(pair_neg * neg_samples, pos_pair_size * i)
    # sys.exit(1)

    anchor, pos, uniq, freq = generate_positive_pairs(X, y, rand_samples= rand_samples, pair_len=pos_samples)
    neg = generate_negative_pairs(X, y, uniq, freq, rand_samples= rand_samples, pair_len=pos_samples)
    
    return anchor, pos, neg



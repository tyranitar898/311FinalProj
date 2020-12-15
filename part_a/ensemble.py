from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random


def sample_with_replacement(data, num_sample_pts):
    result = []
    # num_sample_pts = number of data
    # data is dictionary
    n = len(data['user_id'])
    for i in range(n):
        ran = random.randint(0, n-1)
        this_user_dict = {
            'user_id': data['user_id'][ran],
            'question_id': data['question_id'][ran],
            'is_correct': data['is_correct'][ran],
        }
        result.append(this_user_dict)

    return result


if __name__ == "__main__":
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    bag1 = []
    bag2 = []
    bag3 = []
    bags = [bag1, bag2, bag3]
    n = len(train_data["is_correct"])

    # BAGGING
    for k in range(3):
        bag = sample_with_replacement(train_data, n)
        bags[k].append(bag)


    #TODO: calcualte avg
    print('done')

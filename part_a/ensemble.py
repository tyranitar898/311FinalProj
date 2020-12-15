from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random
from knn import knn_impute_by_user
from item_response import irt, evaluate


def sample_with_replacement(data, num_sample_pts):

    # num_sample_pts = number of data
    # data is dictionary
    n = len(data['user_id'])
    result = {
        'user_id': [],
        'question_id': [],
        'is_correct': [],
    }
    for i in range(n):
        ran = random.randint(0, n-1)
        cur_user_id = data["user_id"][ran]
        cur_question_id = data["question_id"][ran]
        cur_is_correct = data["is_correct"][ran]
        result['user_id'].append(cur_user_id)
        result['question_id'].append(cur_question_id)
        result['is_correct'].append(cur_is_correct)

    return result

def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    bags = []
    n = len(train_data["is_correct"])

    # BAGGING
    for k in range(3):
        bag = sample_with_replacement(train_data, n)
        bags.append(bag)

    train_ll, val_ll, theta1, beta1, val_acc_lst1 = irt(bags[0], val_data, 0.01, 9)
    train_ll, val_ll, theta2, beta2, val_acc_lst2 = irt(bags[1], val_data, 0.01, 9)
    train_ll, val_ll, theta3, beta3, val_acc_lst3 = irt(bags[2], val_data, 0.01, 9)

    print(val_acc_lst1[-1], val_acc_lst2[-1], val_acc_lst3[-1])
    avg_theta = theta1 + theta2 + theta3 / 3
    avg_beta = beta1 + beta2 + beta3 / 3
    print(evaluate(val_data, avg_theta, avg_beta))


    # acc1 = knn_impute_by_user(bag1, val_data, 11)
    # acc2 = knn_impute_by_user(bag2, val_data, 11)
    # acc3 = knn_impute_by_user(bag3, val_data, 11)
    # print(acc1, acc2, acc3)

if __name__ == "__main__":
    main()


    #TODO: calcualte avg
    print('done')

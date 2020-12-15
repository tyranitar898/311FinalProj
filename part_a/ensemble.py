from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random
from knn import knn_impute_by_user
from item_response import irt, sigmoid


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

def evaluate(data, theta, beta, m_samples):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []

    for i, q in enumerate(data["question_id"]):
        x = 0
        for m in range(m_samples):
            u = data["user_id"][i]
            x += (theta[m][u] - beta[m][q]).sum()
        x = x/m_samples
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    bags = []
    n = len(train_data["is_correct"])

    # BAGGING
    m = 3
    thetas = []
    betas = []
    accs = []

    for k in range(m):
        bag = sample_with_replacement(train_data, n)
        bags.append(bag)

    for kk in range(m):
        train_ll, val_ll, theta, beta, val_acc_lst = irt(bags[kk], val_data, 0.01, 9)
        thetas.append(theta)
        betas.append(beta)
        accs.append(val_acc_lst[-1])
    # accuracies of individual IRTs
    print(accs)
    # accuracy of taking avg prediction of IRTs
    print(evaluate(val_data, thetas, betas, m))


if __name__ == "__main__":
    main()

    print('done')

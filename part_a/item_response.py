from utils import *

import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    count = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        count += 1

        if data["is_correct"][i] != np.nan:
            x_i = data["is_correct"][i]
            cur_theta = (theta[cur_user_id])
            cur_beta = (beta[cur_question_id])
            prob = (np.exp(cur_theta - cur_beta))/(1+(np.exp(cur_theta-cur_beta)))

            this_data_likelihood = (x_i*prob + (1-x_i)*(1-prob))
            log_lklihood = log_lklihood + this_data_likelihood
            # print("data_id:{} cur_user_id:{} cur_q_id:{} this_data_likelihood {}".format(i, cur_user_id, cur_question_id, this_data_likelihood))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -np.log(log_lklihood)


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        x = np.exp((theta[cur_user_id]))
        y = np.exp((beta[cur_question_id]))
        if (data["is_correct"][i]):
            theta[cur_user_id] += (lr * (y/(x+y)))
            x = np.exp((theta[cur_user_id]))
            beta[cur_question_id] += (lr * (-y/(x+y)))
        if (not data["is_correct"][i]):
            theta[cur_user_id] += (lr * (-x / (x + y)))
            x = np.exp((theta[cur_user_id]))
            beta[cur_question_id] += (lr * (x / (x + y)))




    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # theta = None
    # beta = None
    # theta = np.random.rand(542,)
    # beta = np.random.rand(1774,)

    theta = np.zeros((542, ))
    beta = np.zeros((1774, ))
    # theta = np.ones(542, )
    # beta = np.ones(1774, )

    val_acc_lst = []
    train_ll = []
    val_ll = []

    for i in range(iterations):

        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_valid = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_ll.append(neg_lld)
        val_ll.append(neg_lld_valid)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("iteration:{} : neg_lld {} \t Score: {}".format(i, neg_lld, score))

        theta, beta = update_theta_beta(data, lr, theta, beta)
        # if i > 1 and (val_acc_lst[i] < val_acc_lst[i-1]):
        #     print("val acc dip @{}".format(i))

    # TODO: You may change the return values to achieve what you want.
    return train_ll, val_ll, theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def prob(theta, beta):
    probb = np.exp(theta - beta)/(1+np.exp(theta-beta))
    return probb


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # print(neg_log_likelihood(train_data, theta, beta))
    # update_theta_beta(train_data, 0.1, theta, beta)
    iterations = 9
    learning_rate = 0.01

    train_ll, val_ll, theta, beta, val_acc_lst = irt(train_data, val_data, learning_rate, iterations)
    iteration_arr = list(range(iterations))
    # plt.title("iteration vs -loglikelihood hyperparamters: lr = {} iterations = {}".format(learning_rate, iteratons))
    #
    # plt.plot(iteration_arr, train_ll, label='train lld')
    # plt.plot(iteration_arr, val_ll, label='valid lld')
    #
    # plt.legend()
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################

    # hyperparameters choose from above
    # iteratons = 9
    # learning_rate = 0.01
    # train_ll, test_ll, theta, beta, val_acc_lst = irt(train_data, test_data, learning_rate, iterations)

    # Part (d)
    theta_arr = list(range(-5, 6))


    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []

    for i in theta_arr:
        y1.append(prob(i, beta=beta[11]))
        y2.append(prob(i, beta=beta[22]))
        y3.append(prob(i, beta=beta[33]))
        y4.append(prob(i, beta=beta[44]))
        y5.append(prob(i, beta=beta[55]))
    plt.title("theta vs p(): ")

    plt.plot(theta_arr, y1, label='q_id 11')
    plt.plot(theta_arr, y2, label='q_id 22')
    plt.plot(theta_arr, y3, label='q_id 33')
    plt.plot(theta_arr, y4, label='q_id 44')
    plt.plot(theta_arr, y5, label='q_id 55')

    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

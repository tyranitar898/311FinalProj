from utils import *

import numpy as np

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
            cur_theta = sigmoid(theta[cur_user_id])
            cur_beta = sigmoid(beta[cur_question_id])
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
    NC = 0
    NIC = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        x = np.exp(sigmoid(beta[cur_question_id]))
        y = np.exp(sigmoid(theta[cur_user_id]))
        if (data["is_correct"][i]):

            theta[cur_user_id] += (lr * (y/(x+y)))
            beta[cur_question_id] += (lr * (-y/(x+y)))
        if (not data["is_correct"][i]):
            theta[cur_user_id] += (lr * (-x / (x + y)))
            beta[cur_question_id] += (lr * (x / (x + y)))
        # if(data["is_correct"][i]):
        #     theta[cur_user_id] += lr * np.exp(beta[cur_question_id])/(np.exp(beta[cur_question_id]) + np.exp(theta[cur_user_id]))
        #     beta[cur_question_id] += -1 *lr * np.exp(beta[cur_question_id])/(np.exp(beta[cur_question_id] + np.exp(theta)))
        # if(not data["is_correct"][i]):
        #     theta[cur_user_id] += lr * -1 * np.exp(theta[cur_user_id]) / (np.exp(beta[cur_question_id]) + np.exp(theta[cur_user_id]))
        #     beta[cur_question_id] += lr* np.exp(beta[cur_question_id]) / (
        #                 np.exp(beta[cur_question_id]) + np.exp(theta[cur_user_id]))

    # for i in range(theta.shape[0]):
    #     top = np.exp(theta[i]) - (np.sum(np.exp(beta))/1)
    #     bot = np.exp(theta[i]) + (np.sum(np.exp(beta))/1)
    #     theta[i] += lr * (-top / bot)
    #
    # for k in range(beta.shape[0]):
    #     top = np.exp(beta[k]) - (np.sum(np.exp(theta))/1)
    #     bot = np.exp(beta[k]) + (np.sum(np.exp(theta))/1)
    #     beta[k] += lr * (-top / bot)

    # for i in range(theta.shape[0]):
    #     term1 = 1 * np.sum(np.exp(beta))/(np.sum(np.exp(beta))+np.sum(np.exp(theta[i])))
    #     term2 = -1 * (np.exp(theta[i]))/(np.sum(np.exp(beta))+np.exp(theta[i]))
    #     theta[i] += lr * (term1 + term2)
    #
    # for k in range(beta.shape[0]):
    #     term1 = -1 * np.sum(np.exp(beta[k]))/(np.sum(np.exp(beta[k]))+np.sum(np.exp(theta)))
    #     term2 = 1 * (np.sum(np.exp(theta)))/(np.sum(np.exp(beta[k]))+np.sum(np.exp(theta)))
    #     beta[k] += lr * (term1 + term2)



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
    theta = np.random.rand(542,)
    beta = np.random.rand(1774,)
    # theta = np.zeros((542, ))
    # beta = np.zeros((1774, ))
    # theta = np.ones((542, ))
    # beta = np.ones((1774, ))

    val_acc_lst = []

    for i in range(iterations):

        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("iteration:{} : {} \t Score: {}".format(i, neg_lld, score))

        theta, beta = update_theta_beta(data, lr, theta, beta)


    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


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
    irt(train_data, val_data, 0.01, 90000)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

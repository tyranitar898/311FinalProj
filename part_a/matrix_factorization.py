from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u: np.ndarray, z: np.ndarray):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    m = train_data["question_id"][i]
    u_n = u[n, :]
    z_m = z[m, :]

    u_n = u_n + lr * (c - u_n.dot(z_m)) * z_m
    z_m = z_m + lr * (c - u_n.dot(z_m)) * u_n

    u[n, :] = u_n
    z[m, :] = z_m

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    losses = []
    for i in range(num_iteration):
        if i % (num_iteration // 100) == 0:
            print(f"{100 * (i / num_iteration)}%")
            losses.append((i, squared_error_loss(train_data, u, z)))
        u, z = update_u_z(train_data, lr, u, z)

    mat = u @ z.transpose()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, losses


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # best k = 9
    # scores = []
    # for k in range(1, 25):
    #     x = svd_reconstruct(train_matrix, k)
    #     scores.append(sparse_matrix_evaluate(val_data, x))
    #     print(k, sparse_matrix_evaluate(val_data, x))
    # plt.plot(range(1, 25), scores)
    # plt.xlabel("k")
    # plt.ylabel("Validation Score")
    # plt.title("SVD k vs Validation Score")
    # plt.show()
    # k = 9
    # x = svd_reconstruct(train_matrix, k)
    # print('Train, k=9', sparse_matrix_evaluate(train_data, x))
    # print('Validation, k=9', sparse_matrix_evaluate(val_data, x))
    # print('Test, k=9', sparse_matrix_evaluate(test_data, x))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    # best hyperparamters found so far
    lr = 0.01
    k = 100
    num_iter = 800000

    mat, losses = als(train_data, k, lr, num_iter)
    print(f"k={k}, lr={lr}, num_iter={num_iter}")
    print(f"Validation Score={sparse_matrix_evaluate(val_data, mat)}")
    print(f"Test Score={sparse_matrix_evaluate(test_data, mat)}")

    # plt.plot([i[0] for i in losses], [i[1] for i in losses])
    plt.title("Squared Error vs Number of Iterations")
    plt.ylabel("Squared Error")
    plt.xlabel("# Iterations")

    plt.plot([i[0] for i in losses], [i[1] for i in losses])
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

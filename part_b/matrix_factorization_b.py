from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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

    return u, z


def als(train_data, lr, num_iteration, initial_u: np.ndarray, initial_z: np.ndarray):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param initial_u: np.ndarray
    :param initial_z: np.ndarray
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = initial_u.copy()
    z = initial_z.copy()
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


def init_uz(train_data, question_metadata, student_metadata, num_subjects):
    subject_lookup = [arr[1] for arr in
                      sorted(list(zip(question_metadata['question_id'], question_metadata['subject_id'])),
                             key=lambda x: x[0])]

    num_students = len(set(train_data["user_id"]))
    num_questions = len(set(train_data["question_id"]))

    u = np.full((num_students, num_subjects), 0)
    qs_in_cat = np.full((num_students, num_subjects), 0)  # number of questions answered in each category per student

    z = np.full((num_questions, num_subjects), 0.1)

    total_answered = np.full(num_students, 0)
    for sid, qid, corr in zip(train_data['user_id'], train_data['question_id'], train_data['is_correct']):
        u[sid, subject_lookup[qid]] += corr
        qs_in_cat[sid, subject_lookup[qid]] += 1
        total_answered[sid] += 1

    for i, subjs in enumerate(subject_lookup):
        z[i, subjs] = 0.9

    with np.errstate(invalid='ignore'):
        u = u / qs_in_cat
    u[np.isnan(u)] = np.random.uniform(0, 1 / np.sqrt(num_subjects), u[np.isnan(u)].shape)
    return u, z


def main():
    num_subjects = 388
    student_meta, stu_heads = load_meta("../data/student_meta.csv", [int, int, datetime.fromisoformat, float])
    question_meta, q_heads = load_meta("../data/question_meta.csv",
                                       [int, lambda x: [int(i) for i in x[1:-1].split(', ')]])
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # # best hyperparamters found so far
    lr = 0.01
    k = 100
    num_iter = 800000
    u, z = init_uz(train_data, question_meta, student_meta, num_subjects)
    mat, losses = als(train_data, lr, num_iter, u, z)
    print(f"k={k}, lr={lr}, num_iter={num_iter}")
    print(f"Validation Score={sparse_matrix_evaluate(val_data, mat)}")
    print(f"Test Score={sparse_matrix_evaluate(test_data, mat)}")

    # plt.plot([i[0] for i in losses], [i[1] for i in losses])
    plt.title("Squared Error vs Number of Iterations")
    plt.ylabel("Squared Error")
    plt.xlabel("# Iterations")

    plt.plot([i[0] for i in losses], [i[1] for i in losses])
    plt.show()


if __name__ == "__main__":
    main()

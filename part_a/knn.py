from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("User Impute with k = {} Validation Accuracy: {}".format(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    #mat.shape == (1774, 542)
    total_prediction = 0
    total_accurate = 0
    for i in range(len(valid_data["is_correct"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]

        if mat[cur_question_id, cur_user_id] >= 0.5 and valid_data["is_correct"][i]:
            total_accurate += 1
        if mat[cur_question_id, cur_user_id] < 0.5 and not valid_data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    acc = total_accurate / float(total_prediction)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    print("Item Impute with k = {} Validation Accuracy: {}".format(k, acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    ks = [1,6,11,16,21,26]
    user_acc = []
    item_acc = []

    # print("--user--")
    # for k in ks:
    #     user_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))
    # plt.title("k vs accuracy on validation using knn impute by user")
    # plt.plot(ks, user_acc)

    # k* = 11
    knn_impute_by_user(sparse_matrix, test_data, 11)

    # print("--item--")
    # for k in ks:
    #     item_acc.append(knn_impute_by_item(sparse_matrix, val_data, k))
    # plt.show()
    # plt.title("k vs accuracy on validation using knn impute by item")
    # plt.plot(ks, item_acc)
    # plt.show()

    # k* = 21
    knn_impute_by_item(sparse_matrix, test_data, 21)


    print("--test--")
    for k in ks:
        print("---")
        user_acc.append(knn_impute_by_user(sparse_matrix, test_data, k))
        item_acc.append(knn_impute_by_item(sparse_matrix, test_data, k))
    plt.title("k vs accuracy on test using knn impute")
    plt.plot(ks, user_acc, label='impute by user')
    plt.plot(ks, item_acc, label='impute by item')
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

import numpy as np


def calc_user_globals(user_item):
    m, n = user_item.shape

    user_globals = {}

    for i in range(m):
        user_globals[i] = {"avg": np.mean(user_item[i]), "std": np.std(user_item[i])}

    return user_globals


def URP(user_globals_1, user_globals_2):
    avgu1 = user_globals_1["avg"]
    avgu2 = user_globals_2["avg"]
    SDu1 = user_globals_1["std"]
    SDu2 = user_globals_2["std"]

    urp = float(abs(avgu1 - avgu2))
    temp2 = float(abs(SDu1 - SDu2))
    temp4 = float((float((urp)) + (temp2)))
    temp3 = 1 + np.exp(temp4)
    urp = float((1 - (1 / temp3)))
    return urp


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    # In the paper this is Table 2.
    user_item = np.array(
        [
            [4, 3, 5, 4, 2],
            [5, 1, 0, 0, 4],
            [4, 2, 1, 2, 1],
            [2, 1, 0, 1, 2],
            [4, 2, 2, 0, 2],
        ]
    )

    # This is transcribed from Table 3. It is the same
    # matrix that is labeled URP
    urp_paper = np.array(
        [
            [0, 0.386, 0.306, 0.403, 0.5],
            [0, 0, 0.304, 0.433, 0.476],
            [0, 0, 0, 0.462, 0.416],
            [0, 0, 0, 0, 0.491],
            [0, 0, 0, 0, 0],
        ]
    )

    # This takes the ratings of each users and calculates the
    # mean and standard deviation for each user
    # Example:
    #   user M: 4, 3, 5, 4, 2
    #   user mean: 3.8
    #   user SD: 0.979
    user_globals = calc_user_globals(user_item)

    result_urp = np.zeros_like(user_item, dtype=np.float32)

    for i in range(user_item.shape[0]):
        for j in range(user_item.shape[1]):
            result_urp[i, j] = URP(user_globals[i], user_globals[j])

    print(np.triu(result_urp, k=1), "\n")

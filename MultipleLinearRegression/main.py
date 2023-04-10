import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy


def dataset(fish):
    reader = pd.read_csv("Fish.csv")
    reader = reader[reader["Species"].str.contains(f"{fish}")]
    x = reader.iloc[:, 2:].to_numpy()
    y = reader.Weight.values

    return x, y


def compute_cost(x, y, w, b):
    m = x.shape[0]

    total_cost = 0.0

    for i in range(m):
        fw_b_i = numpy.dot(x[i], w) + b
        total_cost += (fw_b_i - y[i])**2

    total_cost = total_cost / (2*m)
    return total_cost


def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = numpy.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (numpy.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err

    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db


def gradient_descent(x, y, w, b, iteration_count, alp):
    j_history = []

    for i in range(iteration_count):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alp * dj_dw
        b = b - alp * dj_db

        if i % math.ceil(iteration_count / 10) == 0:
            j_history.append(compute_cost(x, y, w, b))
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")
    return w, b


if __name__ == '__main__':

    fish_name_list = ["Perch", "Bream", "Roach", "Whitefish", "Parkki", "Pike", "Smelt"]

    w_init = [2, 2, 2, 2, 2]

    while 1:
        user_fish = input(f"Enter a fish name\n{fish_name_list}:").title()
        if user_fish in fish_name_list:
            break
        else:
            print("\nEnter a valid name.\n")

    x_train, y_train = dataset(user_fish)

    iteration = int(input("Iteration count:"))
    alpha = float(input("Alpha:"))

    w_final, b_final = gradient_descent(x_train, y_train, w_init, 2, iteration, alpha)

    m, _ = x_train.shape
    predictions = []
    for i in range(m):
        predictions.append(numpy.dot(x_train[i], w_final) + b_final)


    plt.scatter(range(len(y_train)), y_train, marker="X", c="r", label="Actual Value")
    plt.scatter(range(len(predictions)), predictions, c="b", label="Prediction")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()
    plt.show()

    user_features = input("\nFeatures (Lenght1,Lenght2,Lenght3,Height,Width):").split()

    user_features = list(map(float, user_features))
    user_predicton = (numpy.dot(user_features, w_final) + b_final)

    print(f"A fish that has features you have entered is nearly {user_predicton:.2f} grams")

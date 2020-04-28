import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random


def symm_random_walk(t=20):
    """
    Plots a symmetrical random walk.
    :param t: positive value >0 (int) (default=20)
    :return: None.
            Creates a plot.
    """

    random_walk = np.cumsum([0] + random.choices([-1, 1], k=t - 1))

    plt.xticks(range(0, t + 1))

    plt.plot(np.linspace(0, t, t), random_walk)

    plt.ylabel("M(t)")

    plt.title("Symmetrische random walk")
    plt.show()


def scaled_symm_random_walk(n=100, t=10):
    """
    plots a scaled symmetric random walk
        W^(n)(t) = 1/sqrt(n) M_nt
    with M_nt the random walk at position nt
    :param n: positive value >0 (int) (default=100).
                The scaling factor
    :param t: positive value >0 (int) (default=10)
    :return: None.
            Creates a plot.
    """

    scaled_random_walk = np.cumsum([0] + random.choices([-1, 1], k=n * t - 1)) / (np.sqrt(n))

    line_space = np.linspace(0, t, n * t)

    rc('text', usetex=True)
    plt.plot(line_space, scaled_random_walk)

    plt.ylabel(r"$W^{(" + str(n) + ")}(t)$")

    plt.title("Geschaalde symmetrische random walk")
    plt.show()
    # print(random_walk)


def stochastic_process(t=50):
    """
    Plot an example of a stochastic proces
    :param t: positive value >0 (int) (default=20)
    :return: None.
            Creates a plot.
    """
    # make 2 random walks
    random_walks = [np.cumsum([0] + random.choices([-1, 1], k=t - 1)),
                    np.cumsum([0] + random.choices([-1, 1], k=t - 1))]

    # make stochastic process
    # Start with 1, +1 when 2 times in a row is +1 and -1 when 2 times in a row -1
    stochastic_processes = []
    for random_walk in random_walks:
        process = [1, 1]
        for index in range(2, len(random_walk)):
            if random_walk[index - 2] < random_walk[index] and random_walk[index - 1] < random_walk[index]:
                process.append(process[-1] - 1)
            elif random_walk[index - 2] > random_walk[index] and random_walk[index - 1] > random_walk[index]:
                process.append(process[-1] + 1)
            else:
                process.append(process[-1])

            # if index >= 3 and random_walk[index-2] < random_walk[index] and random_walk[index-1] < random_walk[index] and random_walk[index-3] < random_walk[index]:
            #     process[-1] += 2
        stochastic_processes.append(process)

    # print(random_walks)
    # print(stochastic_processes)

    fig, axs = plt.subplots(2, 2)

    # plot the random walks
    axs[0, 0].plot(np.linspace(0, t, t), random_walks[0])
    axs[0, 1].plot(np.linspace(0, t, t), random_walks[1], color='green')

    axs[0, 0].set_title("Random Walk 1")
    axs[0, 1].set_title("Random Walk 2")

    # plot stochastic process
    axs[1, 0].plot(np.linspace(0, t, t), stochastic_processes[0])
    axs[1, 1].plot(np.linspace(0, t, t), stochastic_processes[1], color='green')

    axs[1, 0].set_title("Aangepast proces 1")
    axs[1, 1].set_title("Aangepast proces 2")

    plt.show()


def plot_activation_functions():
    activations = ["Elu", "Softplus", "Softsign", "Relu", "Tanh", "Sigmoid"]

    def elu(x, alpha=1):
        # return np.maximum(alpha*np.expm1(x), x)
        if x >= 0:
            return x
        else:
            return alpha * np.expm1(x)

    def softplus(x):
        return np.log(np.exp(x) + 1)

    def softsign(x):
        return x / (np.abs(x) + 1)

    def relu(x):
        return np.maximum(0, x)

    def tanh(x):
        return np.tanh(x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    functions = [elu, softplus, softsign, relu, tanh, sigmoid]

    fig, axes = plt.subplots(2, 3)

    for index, (name, func) in enumerate(zip(activations, functions)):
        x = np.linspace(-4, 4, 100)
        if name == "Elu":
            y = []
            for element in x:
                y.append(func(element))
            y = np.array(y)
        else:
            y = func(x)

        row = index // 3
        column = index % 3

        axes[row, column].plot(x, y)
        axes[row, column].set_title(name)
        axes[row, column].set_ylim([-1.5, 2.5])
        axes[row, column].grid(True)

    plt.show()


if __name__ == "__main__":
    print("Start")
    # scaled_symm_random_walk()
    # symm_random_walk()
    # stochastic_process()
    # plot_activation_functions()

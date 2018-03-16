import random
import argparse
from utils import *


class LinearRegression(object):

    def __init__(self, data, target, N, _lambda):
        self.data = data
        self.target = target
        self.degree = N
        self._lambda = _lambda

    def ATA_plus_lambdaI(self):
        def _A_init():
            A = []
            for i in range(len(self.data)):
                tmp = []
                for j in range(self.degree):
                    tmp.append((self.data[i])**(self.degree-j-1))
                A.append(tmp)
            return A

        self.A = _A_init()
        self.AT = transpose_2d(self.A)
        self.ATA = matmul(self.AT, self.A)

        res = [[0] * len(self.ATA) for i in range(len(self.ATA))]
        for i in range(self.degree):
            for j in range(self.degree):
                if i == j:
                    res[i][j] = self.ATA[i][j] + self._lambda
                else:
                    res[i][j] = self.ATA[i][j]

        return res

    def LinearRegression(self):
        P, L, U = LU_decomposition(self.ATA_plus_lambdaI())
        # array_print(P)
        # array_print(L)
        # array_print(U)
        inv = matmul(matmul(inverse(U, 'U'), inverse(L, 'L')), P)   # inverse = U-1 * L-1 * P

        w = matmul(matmul(inv, self.AT), convert_to_2darray(self.target))
        self.weights = [w[i][0] for i in range(len(w))]

        return self.weights

    def NewtonMethod(self, epsilon=1e-6):
        """ Implement Newton's Method to optimize the error function (LSE in this example) """
        x = [random.randint(1,100)*0.1 for i in range(self.degree)]
        count = 1
        while(True):
            print("{} iterations {:.5f}".format(count, self.LSE(x)))
            _2ATAx = multiply(matmul(self.ATA, convert_to_2darray(x)), 2)
            _2ATb = multiply(matmul(self.AT, convert_to_2darray(self.target)), 2)

            first_derivative = matsub(_2ATAx, _2ATb)
            second_derivative = multiply(self.ATA, 2)

            x_new = matsub(convert_to_2darray(x), matmul(inverse(second_derivative), first_derivative))
            x_new_flatten = [x_new[i][0] for i in range(len(x_new))]

            flag = 1
            for i in range(self.degree):
                if abs(x_new_flatten[i] - x[i]) > epsilon:
                    flag = 0
            if flag:
                return x_new_flatten
            else:
                count += 1
                x = x_new_flatten

    def LSE(self, weights):
        """ calculate the least square error """
        y = matmul(self.A, convert_to_2darray(weights))
        error = 0
        for i in range(len(self.data)):
            error += (y[i][0] - self.target[i])**2

        return error / len(self.data)


def ReadFile(filename):
    x, y, rest = [], [], []
    with open(filename) as f:
        lines = f.readlines()

        for i in range(len(lines)):
            line = lines[i].split('\n')[0].split(',')

            if len(line) == 2:  # data
                x.append(float(line[0]))
                y.append(float(line[1]))
            else:
                rest.append(float(line[0]))

    return x, y, int(rest[0]), rest[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='specify the data file', default='data.txt')
    args = parser.parse_args()

    data, target, N, _lambda = ReadFile(args.filename)

    regressor = LinearRegression(data, target, N, _lambda)
    regressor.LinearRegression()
    # print(weights)
    print("------- Linear Regression with regularization -------")
    weights1 = regressor.weights
    print_equation(weights1, N)
    print("LSE: %.5f\n" % regressor.LSE(regressor.weights))

    print("------- Linear Regression using Newton Method -------")
    weights2 = regressor.NewtonMethod()
    print_equation(weights2, N)
    # print(weights2)
    print("LSE: %.5f" % regressor.LSE(weights2))

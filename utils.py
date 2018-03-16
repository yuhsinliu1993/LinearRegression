def transpose_2d(mat):
    """ traspose the 2D-array """
    rows, cols = len(mat), len(mat[0])

    result = []
    for i in range(cols):
        tmp = []
        for j in range(rows):
            tmp.append(mat[j][i])
        result.append(tmp)

    return result


def multiply(A, scalar=1):
    """ element-wisely scale 2d matrix """
    ret = [[0]*len(A[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            ret[i][j] = A[i][j] * scalar

    return ret


def matmul(A, B):
    """ 2d matrix multiplication"""
    row_A, col_A = len(A), len(A[0])
    row_B, col_B = len(B), len(B[0])

    if col_A != row_B:
        raise Exception("Cannot multiply matrix A and B\n")

    result = []
    for i in range(row_A):
        tmp = []
        for j in range(col_B):
            _sum = 0
            for k in range(col_A):
                _sum += A[i][k] * B[k][j]
            tmp.append(_sum)
        result.append(tmp)

    return result


def matsub(A, B):
    """ 2d matrix substraction"""
    ret = [[0]*len(A[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            ret[i][j] = A[i][j] - B[i][j]

    return ret


def pivot_matrix(mat):
    """ return the pivoting matrix for M, used in Doolittle's method """
    identity = [[float(i==j) for i in range(len(mat))] for j in range(len(mat))]

    for j in range(len(mat)):
        row = max(range(j, len(mat)), key=lambda i: abs(mat[i][j]))
        if j != row:
            identity[j], identity[row] = identity[row], identity[j]

    return identity


def LU_decomposition(mat):
    """ Doolittle's method: PA = LU, where P is the pivot matrix of A """
    L = [[0.0]*len(mat) for i in range(len(mat))]
    U = [[0.0]*len(mat) for i in range(len(mat))]

    P = pivot_matrix(mat)
    PA = matmul(P, mat)

    for j in range(len(mat)):
        L[j][j] = 1.0

        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1

        for i in range(j, len(mat)):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]

    return P, L, U


def masked_matrix(mat, i, j):
    """ mask out the row_i and col_j of the given matrix """
    masked = []
    for row in range(0, i):
        tmp = []
        for col in range(0, j):
            tmp.append(mat[row][col])
        for col in range(j+1, len(mat)):
            tmp.append(mat[row][col])
        masked.append(tmp)

    for row in range(i+1, len(mat)):
        tmp = []
        for col in range(0, j):
            tmp.append(mat[row][col])
        for col in range(j+1, len(mat)):
            tmp.append(mat[row][col])
        masked.append(tmp)

    return masked


def determinant_of_triangular_matrix(mat):
    _sum = 1
    N = len(mat)
    for i in range(N):
        for j in range(N):
            if i == j:
                _sum *= mat[i][j]

    return _sum


def determinant(mat):
    # check if the given matrix is square matrix
    if len(mat) != len(mat[0]):
        raise Exception('Not square matrix. Cannot compute determinant.')

    if len(mat) == 2:
        return mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]

    det = 0
    for k in range(len(mat)):
        masked = masked_matrix(mat, 0, k)
        det += ((-1)**k) * mat[0][k] * determinant(masked)

    return det


def inverse_2(L, U):
    # LUM = I ==> LA = I, UM = A
    N = len(L)
    Z = [[0.0]*N for i in range(N)]
    A = [[0.0]*N for i in range(N)]
    M = [[0.0]*N for i in range(N)]

    for i in range(N):
        for j in range(N):  # A[i][j]
            if i < j:  # upper
                A[i][j] = 0
            elif i == j:
                A[i][j] = 1
            elif i==1:
                A[i][0] = -1 * L[i][0]
            elif i==2:
                A[i][0] = -1 * L[i][0] - L[i][1]*A[i]





def adjugate(mat):
    adj = []

    if len(mat) == 2:
        adj.append([mat[1][1], -mat[0][1]])
        adj.append([-mat[1][0], mat[0][0]])
        return adj

    for i in range(len(mat)):
        tmp = []
        for j in range(len(mat)):
            masked = masked_matrix(mat, i, j)
            tmp.append(((-1)**(i+j)) * determinant(masked))
        adj.append(tmp)

    return transpose_2d(adj)


def inverse(mat, is_triangle=False):
    if is_triangle:
        det = determinant_of_triangular_matrix(mat)
    else:
        det = determinant(mat)

    adj = adjugate(mat)

    for i in range(len(mat)):
        for j in range(len(mat)):
            adj[i][j] /= float(det)

    return adj


def convert_to_2darray(mat):
    """ convert 1d array to 2d array """

    if type(mat[0]) != list:  # if the input matrix is 2d-array
        ret = [[mat[i]] for i in range(len(mat))]
        return ret
    else:
        return mat


def print_equation(w, N):
    """ y = w[0]*x^N-1 + w[1]*x^N-2 + ... + w[N-1] """
    if N == 1:
        formula = "y = %d" % w[0]
    elif N == 2:
        if w[1] < 0:
            formula = "y = %.5fx - %.5f" % (w[0], -w[1])
        else:
            formula = "y = %.5fx + %.5f" % (w[0], w[1])
    else:
        formula = "y = %.5fx^%d" % (w[0], N-1)
        for i in range(1, N):
            if i == N-1:
                if w[i] < 0:
                    formula += " - %.5f" % -w[i]
                else:
                    formula += " + %.5f" % w[i]
            elif i == N-2:
                if w[i] < 0:
                    formula += " - %.5fx" % -w[i]
                else:
                    formula += " + %.5fx" % w[i]
            else:
                if w[i] < 0:
                    formula += " - %.5fx^%d" % (-w[i], N-1-i)
                else:
                    formula += " + %.5fx^%d" % (w[i], N-1-i)

    print(formula)


def array_print(mat):
    s = 'Array(['
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if i == (len(mat)-1) and j == (len(mat[0])-1):
                s += '{:.5f}'.format(mat[i][j])
            else:
                s += '{:.5f}, '.format(mat[i][j])
        if i < (len(mat)-1):
            s += '\n       '

    s += '])\n'
    print(s)

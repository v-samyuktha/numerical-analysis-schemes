# This program covers the following matrix schemes: 
# 1. Gauss Elimination
# 2. LU Decomposition (Doolitte)
# 3. Inverse of a Matrix
# 4. Gauss-Jacobi Method
# 5. Gauss-Seidel Method
# 6. Given's Method
# 7. Power Method

#Note: givensMethod2() uses an inbuilt function for QR factorization, use this function if givensMethod() fails to give correct solution

#Edit global parameters here
MAX_ITER = 10000
TOLERANCE = 0.0001

import numpy as np

def readMatrixA(square = True):
    if square == False:
        print("Enter number of rows:")
        m = int(input())
        print("Enter number of columns:")
        n = int(input())
    else:
        print("Enter order of the matrix/no. of eqns:")
        n = int(input())
        m = n
    print("Enter the elements of matrix A row-wise (seperated by spaces):")
    A = []
    for i in range(m):
        print("Row", i+1, ":")
        temp = []
        temp2 = input().split(" ")
        if len(temp2)!=n:
            raise Exception("Number of elements entered doesn't match the number of columns specified.")
        for k in range(len(temp2)):
            temp.append(float(temp2[k]))
        A.append(temp)
    print("Matrix A:")
    print(np.matrix(A))
    return A

def readMatrixB(n):
    print("Enter all elements of B (seperated by a space):")
    B = []
    temp = input().split(" ")
    if len(temp)!=n:
        raise Exception("Invalid number of elements.")
    for item in temp:
        B.append(float(item))
    print("Matrix B:")
    print(np.transpose(np.matrix(B)))
    return B

def swapRow(A, row1, row2):
    for k in range(len(A[0])):
        temp = A[row1][k]
        A[row1][k] = A[row2][k]
        A[row2][k] = temp

def forwardElim(A):
    for k in range(len(A)):
        i_max = k
        v_max = A[i_max][k]

        for i in range(k+1,len(A)):
            if(abs(A[i][k])>v_max):
                v_max = A[i][k]
                i_max = i
        if A[k][i_max]==0:
            return k #Singular Matrix
        if (i_max!=k):
            swapRow(A,k,i_max)
        
        for i in range(k+1,len(A)):
            f = A[i][k]/A[k][k]
            for j in range(k+1,len(A[0])):
                A[i][j] -= A[k][j]*f
            A[i][k]=0
        #print(np.matrix(A))

    return -1

def gaussElim(A):
    singular_row = forwardElim(A)
    if singular_row != -1:
        print("Singular Matrix")
        return
    print("Echelon Form:")
    print(np.matrix(A))

def luDecomp(A):
    n = len(A)
    L = [[0 for x in range(n)]
             for y in range(n)]
    U = [[0 for x in range(n)]
             for y in range(n)]
 
    for i in range(n):
        for k in range(i, n):
            # Summation of L(i, j) * U(j, k)
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            # Evaluating U(i, k)
            U[i][k] = A[i][k] - sum
 
        for k in range(i, n):
            if (i == k):
                L[i][i] = 1  # Doolittle
            else:
                # Summation of L(k, j) * U(j, i)
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                if(U[i][i]==0):
                    print("Pivot element becomes zero, hence decomposition not possible without changing original matrix.")
                    return None
                # Evaluating L(k, i)
                L[k][i] = int((A[k][i] - sum) /
                                  U[i][i])
    print("Lower Triangular:")
    print(np.matrix(L))
    print("Upper Triangular:")
    print(np.matrix(U))

def inverse(A, tol=TOLERANCE):
    
    A = np.array(A)
    if np.linalg.det(A) == 0:
        print("The given matrix is singular.")
        return None
    
    n = len(A)
    AM = np.copy(A)
    I = np.identity(n)
    IM = np.copy(I)
 
    indices = list(range(n))
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n): 
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
    
        for i in indices[0:fd] + indices[fd+1:]: 
        
            crScaler = AM[i][fd]
            for j in range(n): 
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
 
    if np.linalg.norm(I-(A@IM)) < tol:
            print("Inverse Matrix: ")
            print(np.matrix(IM))
    else:
        raise ArithmeticError("Matrix inverse out of tolerance.")

def gaussJacobi(A,b,max_iter = MAX_ITER, tol = TOLERANCE):
    n = np.size(A[0])
    b = np.reshape(b,[n,1])
    z = np.reshape(np.diagonal(A),[n,1])
    # i != j
    B = A - np.diag(np.diagonal(A))
    x_o = np.zeros_like(b)
    for i in range(max_iter):
        x = (b - np.dot(B,x_o))/z
        e = np.linalg.norm(x-x_o, ord='fro')
        # print(x,e)
        if e < tol:
            print(f'Iterations converged after {i} steps')
            break
        x_o = x
    print(np.matrix(x))

def gaussSeidel(A,b,max_iter = MAX_ITER,tol = TOLERANCE):
    n = np.size(A[0])
    b = np.reshape(b,[n,1])
    z = np.reshape(np.diagonal(A),[n,1])
    x   = np.zeros_like(b,dtype=np.double)
    for i in range(max_iter):
        x_o = x.copy()
        for j in range(n):
            x[j] = b[j] - np.dot(A[j,:j],x[:j])- np.dot(A[j,j+1:],x_o[j+1:])
            x[j] = x[j]/A[j,j]
        #print(f'{np.linal.norm(x-x_o)}')    
        if np.linalg.norm(x-x_o) < tol:
                print(f'Iterations converged after {i} steps')
                # print(f'{(x-x_o)}')
                break
    print(np.matrix(x))

def checkDiagonal(A, tol = TOLERANCE):
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                continue
            else:
                if abs(A[i][j]) > tol:
                    return False
    return True

def qrFactorization(A):
    temp = A
    i = 0
    while(True):
        Q,R = np.linalg.qr(temp)
        temp = np.dot(R, Q)
        if(checkDiagonal(temp)):
            print("Number of Factorizations: " + str(i+1))
            break
        else:
            i += 1
    return temp

def givensMethod2(x, tol = TOLERANCE):
    A = qrFactorization(x)
    count = 1
    for i in range(len(A)):
        for j in range(len(A[i])):
            if(i == j):
                temp = A[i][j]
                if(abs(temp) < tol):
                    temp = 0
                print("Eigenvalue "+str(count) +": " + str(temp))
                count += 1

def QR_Decomposition(A):
    A = np.array(A)
    n, m = A.shape

    Q = np.empty((n, n))
    u = np.empty((n, n))

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] 

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) 

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R

def givensMethod(A, max_iter=MAX_ITER, tol = TOLERANCE):
    A = np.array(A)
    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < max_iter):
        A_old[:, :] = A_new
        Q, R = QR_Decomposition(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1
    print("Matrix Q:", Q, "Matrix R", R, sep = '\n')

    eigvals = np.diag(A_new)
    print("Eigenvalues: ", eigvals)

def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n

def powerMethod(A, x, max_iter = MAX_ITER, tol = TOLERANCE):
    A = np.array(A)
    x = np.array(x)
    for i in range(max_iter):
        x = np.dot(A, x)
        if i!=0:
            old_lambda = lambda_1
        lambda_1, x = normalize(x)
        steps = i
        if(i!=0 and abs(old_lambda - lambda_1)<tol):
            break
    print("Steps for convergence: ", steps)
    print('Dominant Eigenvalue:', lambda_1)
    print('Corresponding Eigenvector:', x)

choice = 'y'
while choice == 'y':
    print("Pick an option", "1. Gauss Elimination", "2. LU Decomposition", "3. Find Inverse", "4. Gauss-Jacobi", "5. Gauss-Seidel", "6. Given's Method", "7. Power Method", sep = '\n')
    option = int(input())
    if option == 1:
        A= readMatrixA(square = False)
        gaussElim(A)
    elif option == 2:
        A = readMatrixA()
        luDecomp(A)
    elif option == 3:
        A = readMatrixA()
        inverse(A)
    elif option == 4:
        A = readMatrixA()
        B = readMatrixB(len(A))
        gaussJacobi(A,B)
    elif option == 5:
        A = readMatrixA()
        B = readMatrixB(len(A))
        gaussSeidel(A,B)
    elif option == 6:
        A = readMatrixA(square = False)
        givensMethod(A)
    elif option == 7:
        A = readMatrixA(square = False)
        print("Do you have an initial guess for the eigenvector? (y/n)")
        temp_choice = input()
        x = []

        if temp_choice == 'y':
            print("Enter all elements of the guess vector(seperated by spaces)")
            temp_num = input().split(" ")
            if len(temp_num)!=len(A):
                print("Invalid number of elements")
                continue
            for item in temp_num:
                x.append(float(item))
        else:
            for i in range(len(A)):
                x.append(float(1))    

        powerMethod(A,x)
    else:
        print("Invalid Choice")
    print("Would you like to continue? (y/n)")
    choice = input()
# This program covers the following iterative schemes for finding roots of polynomials: 
# 1. Bisection Method
# 2. Secant Method
# 3. Newton-Raphson Method
# 4. Regula-Falsi Method
# 5. General Iterative Scheme

from logging import raiseExceptions
import numpy as np

def get_function():
    print("Enter degree of the polynomial")
    degree = int(input())
    coeffs = []
    print("Enter coefficients as follows:")
    for i in range(degree+1):
        print(f"Coefficient of x^{degree - i}:")
        temp = int(input())
        coeffs.append(temp)
    return np.poly1d(coeffs)
    
def bisectionMethod(f, a, b, tol = 0.1):
    # |f(m)| < tol would be an approximation to the root
    
    # Checking the interval for a root
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("There lies no root between the given endpoints")
        return None
        
    # Updating the midpoint
    mid = (a + b)/2
    
    if np.abs(f(mid)) < tol:
        #Checking for a root with the given tolerance
        return mid

    elif np.sign(f(a)) == np.sign(f(mid)):
        # Changing the interval to (mid,b)
        return bisectionMethod(f, mid, b, tol)

    elif np.sign(f(b)) == np.sign(f(mid)):
        # Changing the interval to (a,mid)
        return bisectionMethod(f, a, mid, tol)

def secantMethod(f, x0, x1, tol = 0.1, max_iter = 1000):
    # store y values instead of recomputing them
    fx0 = f(x0)
    fx1 = f(x1)

    # iterate up to maximum number of times
    for _ in range(max_iter):
        # see whether the answer has converged
        if abs(fx1) < tol:
            return x1

        # do calculation
        x2 = (x0 * fx1 - x1 * fx0) / (fx1 - fx0)
        # shift variables (prepare for next loop)

        x0,  x1  = x1,  x2
        fx0, fx1 = fx1, f(x2)
        
    raise Exception("Secant method has failed to converge")
    return None

def newtonRaphson(f, x_guess, tol = 0.1, max_iter = 1000):
    Dfn = np.polyder(f, 1)
    x0 = x_guess
    xn = x0
    for n in range(0,max_iter+1):
        fxn = f(xn)
        if abs(fxn)<tol:
            print(f"No of iterations for solution = {n}")
            return xn
        Dfxn = Dfn(xn)
        if Dfxn==0:
            print(f"Derivative zero, No solution")
            return None
        xn = xn - fxn/Dfxn
    raise Exception("Newton-Raphson method has failed to converge")
    return None

def regulaFalsi(f, a, b, tol = 0.1, max_iter = 1000):
    if f(a) * f(b) >= 0:
        raiseExceptions("Root doesn't exist in the given interval")
        return None
     
    c = a 
    for i in range(max_iter):
        c = (a * f(b) - b * f(a))/ (f(b) - f(a))
         
        # Check for root
        if np.abs(f(c)) < tol:
            break
         
        # Changing the interval
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c

def generalScheme(f, g, x0, tol = 0.1, max_iter = 1000):
    step = 1
    flag = 1
    condition = True
    while condition:
        x1 = g(x0)
        x0 = x1
        step = step + 1
        
        if step > max_iter:
            flag = 0
            break
        
        condition = abs(f(x1)) > tol

    if flag==1:
        return x1
    else:
        raise Exception('Not Convergent')
        return None

methods = ["Bisection Method", "Secant Method", "Newton-Raphson Method", "Regula-Falsi Method", "General Iterative Scheme"]
functions = [bisectionMethod, secantMethod, newtonRaphson, regulaFalsi, generalScheme]

#Function to display the menu
def printMenu():
    print("Pick an iterative scheme:")
    for i in range(len(methods)):
        print(i+1, '.', methods[i])

f = 'y'
while f=='y':
    printMenu()
    option = input()
    if option == '1':
        f = get_function()
        print("To find a root in the interval [a,b],")
        print("Enter the value of a:")
        a = int(input())
        print("Enter the value of b:")
        b = int(input())
        #print(f"f(a) = {f(a)}")
        #print(f"f(b) = {f(b)}")
        print(f"The root is: {bisectionMethod(f, a, b)}")

    elif option == '2':
        f = get_function()
        print("To find a root in the interval [a,b],")
        print("Enter the value of a:")
        a = int(input())
        print("Enter the value of b:")
        b = int(input())
        print(f"The root is: {secantMethod(f, a, b)}")

    elif option == '3':
        f = get_function()
        print("Enter your guess for the root:")
        x_guess = int(input())
        print(f"The root is {newtonRaphson(f, x_guess)}")

    elif option == '4':
        f = get_function()

        print("To find a root in the interval [a,b],")
        print("Enter the value of a:")
        a = int(input())
        print("Enter the value of b:")
        b = int(input())
        print(f"The root is: {regulaFalsi(f, a, b)}")

    elif option == '5':
        print("Considering the example of f(x) = x^2 - x with an initial guess of 1:")
        f = lambda x: x**2 - x
        g = lambda x: x**2
        x_guess = 1
        print(f"The root is {generalScheme(f, g, x_guess)}")

    else:
        print("Invalid Option")
    print('Do you wish to continue? (y/n)')
    f = input()
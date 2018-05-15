using PyPlot

# Syntax: x = lsgd(A, b, mu, x0, nIters)
#
# Inputs: A is an m x n matrix
# b is a vector of length m
# mu is the step size to use, and must satisfy
# 0 < mu < 2 / norm(A)^2 to guarantee convergence
# x0 is the initial starting vector (of length n) to use
# nIters is the number of iterations to perform
#
# Outputs: x is a vector of length n containing the approximate solution
#
# Description: Performs gradient descent to solve the least squares problem
#
# \min x \|b - A x\|_2
function lsgd(A, b, mu, x0, nIters)
    x = x0
    for i = 1:nIters
        x = x - mu * A' * (A * x - b)
    end
    return x
end

function graph_lsgd()
    m = 100; n = 50; sigma = 0.1

    srand(0) # seed random number generator
    xtrue = rand(n)
    A = randn(m, n)
    b = A * xtrue + sigma * randn(m)
    mu = 0.5 / norm(A)^2
    x0 = rand(n)

    L = 475
    k = zeros(L, 1)
    y = zeros(L, 1)

    for i = 1 : L
        nIters = 24 + i
        k[i] = nIters
        x_k = lsgd(A, b, mu, x0, nIters)
        y[i] = norm(xtrue - x_k)
    end

    # println("k:")
    # println(k)
    # println("y: ")
    # println(y)

    plot(k, y)
    title(string("LSGD Plot with k from 25 to 499 (sigma = ", string(sigma), ")"))
end

print("STARTING...")

graph_lsgd()

print("DONE.")

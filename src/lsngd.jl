# using PyPlot

# Syntax: x = lsngd(A, b, mu, x0, nIters)
#
# Inputs: A is an m x n matrix
# b is a vector of length m
# mu is the step size to use, and must satisfy
# 0 < mu < 1 / norm(A)^2 to guarantee convergence
# x0 is the initial starting vector (of length n) to use
# nIters is the number of iterations to perform
#
# Outputs: x is a vector of length n containing the approximate solution
#
# Description: Performs Nesterov-accelerated gradient descent to solve
# the least squares problem
#
# \min x \|b - A x\|_2
function lsngd(A, b, mu, x0, nIters)
    t_k = 0
    t_k1 = 0

    z_k1 = x0

    x_k = x0
    x_k1 = x0

    for i = 1:nIters
        t_k1 = (1 + (1 + 4 * t_k^2)^0.5) / 2
        z_k1 = x_k1 + (t_k - 1) / t_k1 * (x_k1 - x_k)
        x_k = x_k1
        x_k1 = z_k1 - mu * A' * (A * z_k1 - b)
        t_k = t_k1
    end

    return x_k1
end

function graph_lsngd()
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
        x_k = lsngd(A, b, mu, x0, nIters)
        y[i] = norm(xtrue - x_k)
    end

    # println("k:")
    # println(k)
    # println("y: ")
    # println(y)

    plot(k, y)
    title(string("LSNGD Plot with k from 25 to 499 (sigma = ", string(sigma), ")"))
end

print("STARTING...")

graph_lsngd()

print("DONE.")

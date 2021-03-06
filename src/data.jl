using Distributions
using Polynomials
using PyPlot

#########################################
####### Data Generating Functions #######
#########################################

# Return A and b for least square problems, where:
#   - A: a matrix whose dimension is specified by the tuple dim_data,
#        and whose data is generated from a normal distribution
#        specified by mu and sigma
#   - b: a vector of length dim_data[1],
#        generated by multiplying A with a random vector
#        consisting of values in [0, 1] and of length dim_data[2].
#        Note that the values of b contain random errors within +/-
#        error_percentage of their true values.
function generate_gaussian_data(dim_data, mu, sigma, error_percentage=0.1)
    d = Normal(mu, sigma)
    A = rand(d, dim_data[1], dim_data[2])
    xtrue = rand(dim_data[2])
    b = A * xtrue
    for i = 1 : dim_data[1]
        error = 2 * (rand() - 0.5) * error_percentage
        b[i] = b[i] * (1 + error)
    end
    return A, b, xtrue
end

# Generate a random polynomial of num_degree and num_data number of num_data
# from this polynomial whose error is bounded by the specified error_percentage.
# More specifically:
#   Precondition: - num_degree: integer >= 0
#                 - num_data: integer >= 1
#                 - error_percentage: 1 > float > 0
#   Returns: - A: matrix of dimension num_data by (num_degree + 1),
#                 whose rows consist of x^0, x^1, x^2, ... , x^(num_degree),
#                 where x is a random number in [0, 1] generated by rand()
#            - b: vector of length num_data, whose elements are the values of
#                 the random polynomial evaluated at the randomly generated x's,
#                 in the same order of the rows of A.
#                 Note that the coefficients of the random polynomial are
#                 generated by rand() and therefore in [0, 1].
#                 The values of b also contain random errors within +/-
#                 error_percentage of their true values.
function generate_poly_data(num_data, num_degree, error_percentage=0.1)
    coeffs = zeros(num_degree+1) # from a_0, a_1, ... , a_n
    for i = 1 : num_degree+1
        coeffs[i] = rand()
    end
    p = Poly(coeffs)

    b = zeros(num_data, 1)
    A = zeros(num_data, num_degree + 1)

    for i = 1 : num_data
        x = rand()
        y = p(x)
        error = 2 * (rand() - 0.5) * error_percentage
        y = y * (1 + error)
        b[i] = error
        for j = 1 : num_degree + 1
            A[i, j] = x^(j - 1)
        end
    end

    return A, b, coeffs
end

##########################################
####### Gradient Descent Functions #######
##########################################

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

# TODO: HEAVY-BALL
function lshgd(A, b, mu, x0, nIters)
    t_k = 0
    t_k1 = 0

    z_k1 = x0

    x_k = x0
    x_k1 = x0

    for i = 1:nIters
        t_k1 = (1 + (1 + 4 * t_k^2)^0.5) / 2
        z_k1 = x_k1 + (t_k - 1) / t_k1 * (x_k1 - x_k)
        x_k1 = z_k1 - mu * A' * (A * x_k1 - b)
        x_k = x_k1
        t_k = t_k1
    end

    return x_k1
end

###############################
####### Graph Functions #######
###############################

function graph_lsgd(A, b, xtrue)
    mu = 0.5 / norm(A)^2
    x0 = rand(size(A)[2])

    L = 475
    k = zeros(L, 1)
    y = zeros(L, 1)

    for i = 1 : L
        nIters = 24 + i
        k[i] = nIters
        x_k = lsgd(A, b, mu, x0, nIters)
        y[i] = norm(xtrue - x_k)
    end

    plot(k, y)
    xlabel("Number of Iterations")
    ylabel("Least Sqaures Error")
    # label("LSGD: Standard Gradient Descent")
    title("LSGD Plot with iterations from 25 to 499")
end

function graph_lsngd(A, b, xtrue)
    mu = 0.5 / norm(A)^2
    x0 = rand(size(A)[2])

    L = 475
    k = zeros(L, 1)
    y = zeros(L, 1)

    for i = 1 : L
        nIters = 24 + i
        k[i] = nIters
        x_k = lsngd(A, b, mu, x0, nIters)
        y[i] = norm(xtrue - x_k)
    end

    plot(k, y)
    xlabel("Number of Iterations")
    ylabel("Least Sqaures Error")
    # label("LSNGD: Nesterov-accelerated Gradient Descent")
    title("LSNGD Plot with iterations from 25 to 499")
end

function graph_lshgd(A, b, xtrue)
    mu = 0.5 / norm(A)^2
    x0 = rand(size(A)[2])

    L = 475
    k = zeros(L, 1)
    y = zeros(L, 1)

    for i = 1 : L
        nIters = 24 + i
        k[i] = nIters
        x_k = lshgd(A, b, mu, x0, nIters)
        y[i] = norm(xtrue - x_k)
    end

    plot(k, y)
    xlabel("Number of Iterations")
    ylabel("Least Sqaures Error")
    # label("LSHGD: Heavy-ball Gradient Descent")
    title("LSHGD Plot with iterations from 25 to 499")
end

#############################
####### Run Functions #######
#############################

srand(0) # seed random number generator

print("Generating data...")

A1, b1, xtrue1 = generate_gaussian_data((100, 50), 0, 1, 0.1)

A2, b2, xtrue2 = generate_poly_data(100, 49, 0.1)

print("Generated data. Now doing lsgd...")
figure(1)
graph_lsgd(A1, b1, xtrue1)
# graph_lsgd(A2, b2, xtrue2)

print("Did lsgd. Now doing lsngd.")
figure(2)
graph_lsngd(A1, b1, xtrue1)
# graph_lsngd(A2, b2, xtrue2)

print("Did lsngd. Now doing lshgd.")
figure(3)
graph_lshgd(A1, b1, xtrue1)
# graph_lshgd(A2, b2, xtrue2)
# title("Gradient Descent for Gaussian Data with STD = 1")


print("Done.")

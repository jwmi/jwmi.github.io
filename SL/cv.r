# R code to illustrate cross-validation.
# This code accompanies Lecture 9: Cross-validation.
#
# Author: Jeffrey W. Miller
# Date: Feb 23, 2019

# ____________________________________________________________________________________________

# KNN regression algorithm (for univariate x's)
#   x0 = new point at which to predict y
#   x = (x_1,...,x_n) = vector of training x's
#   y = (y_1,...,y_n) = vector of training y's
#   K = number of neighbors to use
#   y0_hat = predicted value of y at x0
KNN = function(x0, x, y, K) {
    distances = abs(x - x0)  # Euclidean distance between x0 and each x_i
    o = order(distances)  # order of the training points by distance from x0 (nearest to farthest)
    y0_hat = mean(y[o[1:K]])  # take average of the y values of the K nearest training points
    return(y0_hat)  # return predicted value of y
}


# ____________________________________________________________________________________________
# Implementing cross-validation

# Simulate a dataset with univariate x's
set.seed(1)  # set random number generator
n = 100  # number of samples
x = 5*runif(n)  # simulate training x's uniformly on the interval [0,5]
sigma = 0.3  # standard deviation of the noise
f = function(x) { cos(x) }  # f(x) = true mean of x given y
y = f(x) + sigma*rnorm(n)  # simulate training y's by adding N(0, sigma^2) noise to f(x)
plot(x,y,col=2,pch=20,cex=2)  # plot training data


# Using cross-validation to estimate test performance
K = 1  # number of neighbors to use
nfolds = 10  # number of folds to use for cross-validation
permutation = sample(1:n)  # random ordering of all the available data
MSE_fold = rep(0,nfolds)  # vector to hold MSE for each fold
for (j in 1:nfolds) {
    pseudotest = permutation[floor((j-1)*n/nfolds+1) : floor(j*n/nfolds)]  # pseudo-test set for this fold
    pseudotrain = setdiff(1:n, pseudotest)  # pseudo-training set for this fold
    y_hat = sapply(x[pseudotest], function(x0) { KNN(x0, x[pseudotrain], y[pseudotrain], K) })  # run KNN at each x in the pseudo-test set
    MSE_fold[j] = mean((y[pseudotest] - y_hat)^2)  # compute MSE on the pseudo-test set
}
MSE_cv = mean(MSE_fold)  # average across folds to obtain CV estimate of test MSE
MSE_cv


# Compare with a "ground truth" estimate of test performance, given this training set.
# (Since this is a simulation example, we can generate lots of test data.)
n_test = 100000
x_test = 5*runif(n_test)  # simulate test x's from true data generating process
y_test = f(x_test) + sigma*rnorm(n_test)  # simulate test y's from true data generating process
y_test_hat = sapply(x_test, function(x0) { KNN(x0, x, y, K) })  # run KNN at each x in the test set
MSE_test = mean((y_test - y_test_hat)^2)  # compute MSE on test set

MSE_test
MSE_cv

# Careful! Try this with nfolds=n:
sqrt(MSE_test)  # test RMSE
sqrt(mean(MSE_fold))  # sqrt of MSE_cv
mean(sqrt(MSE_fold))  # average of RMSEs (Is this a good way to estimate test RMSE?)


# ____________________________________________________________________________________________
# Using cross-validation to choose model settings

# Illustration: Choosing # of neighbors in KNN
K_max = 30  # maximum value of K to try for KNN
nfolds = 10  # number of folds to use for cross-validation
permutation = sample(1:n)  # random ordering of all the available data
MSE_fold = matrix(0,nfolds,K_max)  # vector to hold MSE for each fold and each K
for (j in 1:nfolds) {
    pseudotest = permutation[floor((j-1)*n/nfolds+1) : floor(j*n/nfolds)]  # pseudo-test set for this fold
    pseudotrain = setdiff(1:n, pseudotest)  # pseudo-training set for this fold
    for (K in 1:K_max) {
        y_hat = sapply(x[pseudotest], function(x0) { KNN(x0, x[pseudotrain], y[pseudotrain], K) })  # run KNN at each x in the pseudo-test set
        MSE_fold[j,K] = mean((y[pseudotest] - y_hat)^2)  # compute MSE on the pseudo-test set
    }
}
MSE_cv = colMeans(MSE_fold)  # average across folds to obtain CV estimate of test MSE for each K

plot(1:K_max, MSE_cv, pch=19)  # plot CV estimate of test MSE for each K

# Choose the value of K that minimizes estimated test MSE
K_cv = which.min(MSE_cv)
K_cv


# ____________________________________________________________________________________________
# Careful: MSE_cv[K_cv] may systematically underestimate or overestimate test MSE!
# There are two sources of bias: K_cv is the minimum, and the pseudo-training set is smaller than n.

# Compare with a "ground truth" estimate of test performance, given this training set.
y_test_hat = sapply(x_test, function(x0) { KNN(x0, x, y, K_cv) })  # run KNN at each x in the test set
MSE_test = mean((y_test - y_test_hat)^2)  # compute MSE on test set

MSE_cv[K_cv]
MSE_test
# Why does MSE_cv[K_cv] tend to be smaller than MSE_test in this example?

# Can you think of a reason why MSE_cv[K_cv] might tend to be larger than MSE_test in other examples?
# (What happens when n=20 and nfolds=2?)


# ____________________________________________________________________________________________
# Choosing the number of folds

# Simulate dataset
set.seed(1)  # set random number generator
n = 20  # number of samples
x = 5*runif(n)  # simulate training x's uniformly on the interval [0,5]
sigma = 0.3  # standard deviation of the noise
y = f(x) + sigma*rnorm(n)  # simulate training y's by adding N(0, sigma^2) noise to f(x)

# Compute "ground truth" estimate of test performance, given this training set
K = 10  # number of neighbors to use in KNN
y_test_hat = sapply(x_test, function(x0) { KNN(x0, x, y, K) })  # run KNN at each x in the test set
MSE_test = mean((y_test - y_test_hat)^2)  # compute MSE on test set

# Repeatedly run CV for a range of nfolds values
nfolds_max = n  # maximum value of nfolds to use for CV
nreps = 200  # number of times to repeat the simulation
MSE_cv = matrix(0,nreps,nfolds_max)  # vector to hold CV estimate of MSE for each rep and each fold
for (r in 1:nreps) {  # run the simulation many times
    for (nfolds in 1:nfolds_max) {
        permutation = sample(1:n)  # random ordering of all the available data
        MSE_fold = rep(0,nfolds)  # vector to hold MSE for each fold and each K
        for (j in 1:nfolds) {
            pseudotest = permutation[floor((j-1)*n/nfolds+1) : floor(j*n/nfolds)]  # pseudo-test set
            pseudotrain = setdiff(1:n, pseudotest)  # pseudo-training set
            y_hat = sapply(x[pseudotest], function(x0) { KNN(x0, x[pseudotrain], y[pseudotrain], K) })  # run KNN at each x in the pseudo-test set
            MSE_fold[j] = mean((y[pseudotest] - y_hat)^2)  # compute MSE on the pseudo-test set
        }
        MSE_cv[r,nfolds] = mean(MSE_fold)
    }
}

# Compute the MSE, bias, and variance of the CV estimate of test MSE, for each value of nfolds
mse = colMeans((MSE_cv - MSE_test)^2)
bias = colMeans(MSE_cv) - MSE_test
variance = apply(MSE_cv,2,var)

# Plot MSE, bias^2, and variance of the CV estimate, for each value of nfolds
plot(1:nfolds_max, type="n", ylim=c(0,max(mse[2:nfolds_max])*1.1), xlab="nfolds", ylab="mse", main="MSE of the CV estimates")
lines(1:nfolds_max, mse, col=1, lty=2, lwd=2, ylim=c(0,0.2))
lines(1:nfolds_max, bias^2, col=2, lwd=2)
lines(1:nfolds_max, variance, col=4, lwd=2)
legend("topright", legend=c("mse","bias^2","variance"), col=c(1,2,4), lwd=2)

# Plot bias for each value of nfolds
plot(1:nfolds_max, bias)
lines(1:nfolds_max, bias, col=2, lwd=2)
# Note that MSE_cv is biased upwards, since the pseudo-training set is smaller than the training set.



























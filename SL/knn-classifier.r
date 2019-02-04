# R code to illustrate K-nearest neighbors (KNN) classifier and the Bayes optimal classifier.
# This code accompanies Lecture 3: Measuring performance.
#
# Author: Jeffrey W. Miller
# Date: Feb 3, 2019

# ____________________________________________________________________________________________

# KNN classifier algorithm (for univariate x's and binary y's) -- probability version
#   x0 = new point at which to predict the y value
#   x = (x_1,...,x_n) = vector of training x's, where x[i] is real-valued
#   y = (y_1,...,y_n) = vector of training y's, where y[i] is 0 or 1
#   K = number of neighbors to use
#   p1_hat = estimated probability of y0=1 given x0
#   Note: We can transform p1_hat to a prediction of the y value at x0 by thresholding p1_hat.
KNN_classifier = function(x0, x, y, K) {
    distances = abs(x - x0)  # Euclidean distance between x0 and each x_i
    o = order(distances)  # order of the training points by distance from x0 (nearest to farthest)
    p1_hat = mean(y[o[1:K]])  # proportion of y values of the K nearest training points that are equal to 1
    return(p1_hat)  # return estimated probability of y0=1
}

# ____________________________________________________________________________________________
# Demonstrate KNN classifier

# Simulate a dataset with univariate x's
set.seed(1)  # set random number generator
n = 20  # number of samples
x = 5*runif(n)  # simulate training x's uniformly on the interval [0,5]
p1 = function(x) { exp(2*cos(x))/(1 + exp(2*cos(x))) }  # p1(x) = true probability of y=1 given x (true relationship between x and y)
y = rbinom(n,1,p1(x))  # simulate training y's as Bernoulli r.v.s with probabilities p1(x)
plot(x,y,col=2,pch=20,cex=2)  # plot training data
x_grid = seq(from=0, to=5, by=0.01)  # grid of x values at which to plot true and predicted y values
lines(x_grid,p1(x_grid))  # plot true p1(x) values for the grid

# Run KNN to predict y at each point on the grid of x values
K = 1  # number of neighbors to use
p1_grid_hat = sapply(x_grid, function(x0) { KNN_classifier(x0, x, y, K) })  # run KNN classifier at each x in the grid
y_grid_hat = round(p1_grid_hat > 0.5)  # predict the y values for each x in the grid by thresholding the estimated probabilities
plot(x,y,col=2,pch=20,cex=2)  # plot training data
title(paste("K =",K))
lines(x_grid,p1(x_grid))  # plot true p1(x) values for the grid
lines(x_grid,p1_grid_hat,col=4)  # plot the estimated probabilities of y=1 for each x0 in the grid
lines(x_grid,y_grid_hat,col=3)  # plot the predicted y values for each x0 in the grid


# ____________________________________________________________________________________________
# Error rates and the Bayes optimal classifier

# Training error rate
p1_hat = sapply(x, function(x0) { KNN_classifier(x0, x, y, K) })  # run KNN classifier (probability version) at each x in the training set
y_hat = round(p1_hat > 0.5)  # predict the y values for each x in the training set (prediction version of KNN)
train_error = mean(y_hat != y)  # compute the training error rate
print(paste0("Training error rate (K = ",K,"): ",train_error))

# Test error rate
n_test = 10000  # large number of samples to simulate as a test set
x_test = 5*runif(n_test)  # simulate test x's
y_test = rbinom(n_test,1,p1(x_test))  # simulate test y's
p1_test_hat = sapply(x_test, function(x0) { KNN_classifier(x0, x, y, K) })  # run KNN classifier (probability version) at each x in the test set
y_test_hat = round(p1_test_hat > 0.5)  # predict the y values for each x in the test set (prediction version of KNN)
test_error = mean(y_test_hat != y_test)  # compute the test error rate
print(paste0("Test error rate (K = ",K,"): ",test_error))
# How can we tell if this is a good test error rate?  Since this is a simulation, we can compare with the best possible test error rate...

# Bayes optimal classifier
y_hat_optimal = round(p1(x) > 0.5)  # use the true p1(x) to make the best possible predictions on the training set
train_error_optimal = mean(y_hat_optimal != y)  # compute the training error rate for the Bayes optimal classifier
print(paste0("Training error rate (Optimal): ",train_error_optimal))
y_test_hat_optimal = round(p1(x_test) > 0.5)  # use the true p1(x) to make the best possible predictions on the test set
test_error_optimal = mean(y_test_hat_optimal != y_test)  # compute the test error rate for the Bayes optimal classifier
print(paste0("Test error rate (Optimal): ",test_error_optimal))

# ____________________________________________________________________________________________

# Bias-variance tradeoff
K = 1  # number of neighbors to use
n_datasets = 50  # number of data sets to simulate, to approximate expectations over the Y's (with fixed x's)
for (i in 1:n_datasets) {
    y = rbinom(n,1,p1(x))  # simulate training y's
    plot(x,y,col=2,pch=20,cex=2,ylim=c(0,1))  # plot training data
    lines(x_grid,p1(x_grid))  # plot true p1(x) values for the grid
    p1_grid_hat = sapply(x_grid, function(x0) { KNN_classifier(x0, x, y, K) })  # run KNN classifier at each x in the grid
    # y_grid_hat = round(p1_grid_hat > 0.5)  # predict the y values for each x
    lines(x_grid,p1_grid_hat,col=4)  # plot predicted p1(x) values for the grid
    # lines(x_grid,y_grid_hat,col=3)  # plot the predicted y values for each x0 in the grid
    Sys.sleep(0.1)  # pause in order to display the plot
    # readline()  # Wait for <Enter> to continue
}







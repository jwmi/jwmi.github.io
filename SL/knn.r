# R code to illustrate K-nearest neighbors (KNN) regression and the bias-variance tradeoff
# This code accompanies Lecture 3: Measuring performance.
#
# Author: Jeffrey W. Miller
# Date: Jan 30, 2019.
# Revised: Jan 31, 2019.  Changes: 0.3 -> sigma, replot points, Sys.sleep(0.1) to refresh plots, clarified comments/names, augment bias-variance illustration.
# Revised: Feb 3, 2019.  Changes: Clarified comments.

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

# Simulate a dataset with univariate x's
set.seed(1)  # set random number generator
n = 20  # number of samples
x = 5*runif(n)  # simulate training x's uniformly on the interval [0,5]
sigma = 0.3  # standard deviation of the noise
f = function(x) { cos(x) }  # f(x) = true mean of x given y
y = f(x) + sigma*rnorm(n)  # simulate training y's by adding N(0, sigma^2) noise to f(x)
plot(x,y,col=2,pch=20,cex=2)  # plot training data
x_grid = seq(from=0, to=5, by=0.01)  # grid of x values at which to plot true and predicted f(x) values
lines(x_grid,f(x_grid))  # plot true f(x) values for the grid

# Run KNN to predict y at each point on the grid of x values
K = 1  # number of neighbors to use
y_grid_hat = sapply(x_grid, function(x0) { KNN(x0, x, y, K) })  # run KNN at each x in the grid
plot(x,y,col=2,pch=20,cex=2)  # plot training data
lines(x_grid,f(x_grid))  # plot true f(x) values for the grid
lines(x_grid,y_grid_hat,col=4)  # plot predicted y values for the grid

# Bias-variance tradeoff
K = 1  # number of neighbors to use
x0 = 1.5  # value of x at which to make a prediction
n_datasets = 10000  # number of data sets to simulate, to approximate expectations over the Y's (with fixed x's)
y0_hat = rep(0,n_datasets)  # initialize vector of zeros to hold predicted y values at x0
y0 = rep(0,n_datasets)  # initialize vector of zeros to hold true y values at x0
for (i in 1:n_datasets) {
    y = f(x) + sigma*rnorm(n)  # simulate training y's
    y0[i] = f(x0) + sigma*rnorm(1)  # simulate true y value at x0
    y0_hat[i] = KNN(x0, x, y, K)  # predicted value of y at x0
    if (i <= 50) {
        plot(x,y,col=2,pch=20,cex=2,ylim=c(-1.5,1.5))  # plot training data
        lines(x_grid,f(x_grid))  # plot true f(x) values for the grid
        y_grid_hat = sapply(x_grid, function(x0) { KNN(x0, x, y, K) })  # run KNN at each x in the grid
        lines(x_grid,y_grid_hat,col=4)  # plot predicted y values for the grid
        points(x0,y0_hat[i],pch=20,cex=4,col=4)  # plot predicted value of y at x0
        Sys.sleep(0.1)  # pause in order to display the plot
        # readline()  # Wait for <Enter> to continue
    }
}
bias = mean(y0_hat) - f(x0)  # bias of KNN predictions at x0
variance = var(y0_hat)  # variance of KNN predictions at x0
noise = sigma^2  # variance of the noise at x0

bias^2 + variance + noise  # bias-variance representation of test MSE at x0
mean((y0_hat - y0)^2)  # test MSE at x0
# Do you get exactly the same number for the preceding two lines?  Does that make sense?  Why or why not?                                            




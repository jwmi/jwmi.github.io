# R code for Lab 1
#
# Instructions: Fill in the correct answer in all of the places where YOUR_ANSWER_HERE appears.
#
# Names of team members
#    Jeffrey W. Miller

# ____________________________________________________________________________________________
# Problems 1 and 2 are to be handed in on paper during class.

# ____________________________________________________________________________________________
# Problem 3

# In lecture, we saw R code for a KNN classifier for univariate x's and binary y's (knn-classifier.r).
# In this part, you will extend this to a KNN classifier for multivariate x's and more than two classes.
#
# Complete the function below.

# KNN classifier algorithm (multivariate, multiclass) -- probability version
#   x0 = length d vector, the new x point at which to predict the y value.
#   x = d-by-n matrix of training x's, where x[,i] is the i'th training point x_i.
#   y = (y[1],...,y[n]) = vector of training y's, where y[i] is in {1,...,C}.
#   K = number of neighbors to use.
#   C = number of classes.
#   p_hat = (p_hat[1],...,p_hat[C]) where p_hat[j] = estimated probability of y0=j given x0.
KNN_multi = function(x0, x, y, K, C) {
    distances = sqrt(colSums((x - x0)^2))  # Euclidean distance between x0 and each x[,i]
    o = order(distances)  # order of the training points by distance from x0 (nearest to farthest)
    p_hat = sapply(1:C, function(j){sum(y[o[1:K]]==j)/K})  # p_hat[j] = proportion of y values of the K nearest training points that are equal to j.
    return(p_hat)  # return estimated probabilities
}

# KNN classifier algorithm (multivariate, multiclass) -- prediction version
KNN_multi_predict = function(x0, x, y, K, C) {
    p_hat = KNN_multi(x0, x, y, K, C)  # compute the estimated probabilities
    y0 = which.max(p_hat)  # find the class with the highest estimated probability
    return(y0)  # return the predicted class
}

# ____________________________________________________________________________________________
# Problem 4

# Simulate training and test data
set.seed(1)  # reset the random number generator
d = 2  # dimension of each training point x_i
n = 100  # number of training samples to simulate
n_test = 10000  # number of test samples to simulate
f = function(x0) { (x0[1]>0)+(x0[2]>0)+1 }  # true relationship between x's and y's
x = matrix(rnorm(n*d),d,n)  # simulate matrix of training x's
y = apply(x,2,f)  # simulate training y's with no noise
x_test = matrix(rnorm(n_test*d),d,n_test)  # simulate test x's
y_test = apply(x_test,2,f)  # simulate test y's
plot(x[1,],x[2,],col=y+1,pch=19)  # plot dimensions 1 and 2

# Apply your KNN classifier with K=5 when (a) d=2 and (b) d=20.  (For d=20, change d above and rerun the code.)
C = 3  # number of classes
K = 5  # number of neighbors to use
y_hat = apply(x, 2, function(x0) { KNN_multi_predict(x0, x, y, K, C) })  # predictions on the training set
y_test_hat = apply(x_test, 2, function(x0) { KNN_multi_predict(x0, x, y, K, C) })  # predictions on the test set
train_error = mean(y_hat != y)  # compute the training error rate
test_error = mean(y_test_hat != y_test)  # compute the test error rate

# Report your results:
# (a) d=2
#    training error rate = 0.05
#    test error rate = 0.1145
#
# (b) d=20
#    training error rate = 0.32
#    test error rate = 0.4572


# ____________________________________________________________________________________________
# Problem 5

# Compute test error when we always naively predict y0=2, regardless of x0.
naive_test_error = mean(2 != y_test)

# Compute test error when we make the optimal prediction f(x0).
optimal_test_error = mean(apply(x_test,2,f) != y_test)

# What do you get for the test error of the naive and optimal prediction methods?
#    naive_test_error = 0.4998
#    optimal_test_error = 0
#
# Should the naive and optimal results depend on whether d=2 or d=20? Explain.
#    No, the naive and optimal predictions depend only on the first two dimensions.
#    There may be slight differences due to randomness in the dataset.
#
# In one or two sentences, compare the performance of KNN when d=2 with KNN when d=20, 
# relative to the naive and optimal performance.
#    The optimal method has a perfect test error of 0, and when d=2, KNN is doing reasonably well
#    with test_error=0.1145.  However, when d=20, KNN is doing much worse with test_error=0.4572, 
#    which is close to the naive performance of 0.4998.


# ____________________________________________________________________________________________
# Problem 6

# Plot the KNN predictions for the first 100 test points when (a) d=2 and (b) d=20.
# (Change d above and rerun the code for each case.)
plot(x_test[1,1:100],x_test[2,1:100],col=y_test_hat[1:100]+1,pch=19)  # show dims 1 and 2 only

# Briefly describe what you see in one or two sentences:
#    When d=2, the plot of predictions looks much like the plot for the true values.
#    However, when d=20, the predictions are much more random looking and scattered.


# ____________________________________________________________________________________________
# Problem 7
# Cancer subtype classification

# Download the following two files from Files/Labs on Canvas:
#    lab-1-gene-expression.txt
#    lab-1-leukemia-type.txt
#
# These files contain gene expression data for n=72 leukemia patients,
# along with the corresponding leukemia subtype (ALL, AML, or MLL).
# 
# This data is from:
#    Armstrong et al., MLL translocations specify a distinct gene expression profile that 
#    distinguishes a unique leukemia. Nature Genetics, 30(1):41, 2002.

# Change directory to where the data files are located.
# (Edit this to wherever you downloaded the data files onto your computer.)
setwd("C:/Users/jwmil/gdrive/SL/Lab-1-KNN")

# Load the gene expression data
Dx = read.table(file="lab-1-gene-expression.txt", header=TRUE, sep='\t', row.names=1)
x = sapply(Dx, as.numeric)

# Load the cancer subtype labels
Dy = read.table(file="lab-1-leukemia-type.txt", header=TRUE, sep='\t', row.names=1)
y = as.numeric(Dy[,1])

# Split into training and test subsets
n = length(y)  # number of samples
set.seed(1)  # reset the random number generator
train = sample(1:n, round(n/2))  # random subset to use as training set
test = setdiff(1:n, train)  # subset to use as test set

# Run your KNN classifier with (a) K=5 and (b) K=30 to make predictions on the test set x[,test]
# based on the training data in x[,train] and y[train].  Compute the error rate on the test set by 
# comparing your predictions with y[test]. Report your results below, using the same train/test split
# for both K=5 and K=30.
C = 3  # number of classes
K = 5  # number of neighbors to use
y_test_hat = apply(x[,test], 2, function(x0) { KNN_multi_predict(x0, x[,train], y[train], K, C) })  # predictions on the test set
test_error = mean(y_test_hat != y[test])  # compute the error rate on the test set

# Report your results:
# (a) K=5
#    test error rate = 0.05555
#
# (b) K=30
#    test error rate = 0.30555











# R code to accompany Lecture 7: Classification, for BST 263 Statistical Learning
#
# Topics:
#    Loss functions and Decision theory
#    Confusion matrix and ROC curves
#    Logistic regression
#    Linear/Quadratic Discriminant Analysis (LDA/QDA)
#
# Author: Jeffrey W. Miller
# Date: Feb 18, 2019

# _____________________________________________________________________________
# Loss functions and Decision theory

# Example: ELISA test for HIV detection

# Simulate data
set.seed(1)  # reset RNG
n = 10000  # number of subjects tested
p_infected = 0.01  # proportion of tested subjects with HIV
y = rbinom(n,1,p_infected)  # simulated HIV status (TRUE: infected, FALSE: not infected)
mu = 0.0  # (try mu=0,1,2,4)  # mu = mean of infected group, 0 = mean of uninfected group
x = rnorm(n) + mu*y  # simulated antigen measurement for n subjects

# Plot the class-conditional distributions
x_grid = seq(from=-4, to=6, by=0.01)
plot(x_grid, 0*x_grid, type="n", ylim=c(0,0.5), xlab="x", ylab="p(x|y)", main="Density of x given y=0 and y=1")
lines(x_grid, dnorm(x_grid,0,1), col=2, lwd=3)
lines(x_grid, dnorm(x_grid,mu,1), col=4, lwd=3)

# Plot data
hist(x, breaks=100)  # histogram of the x's
plot(x, jitter(y), col=2,pch=20,cex=1)  # plot the class y for each x

# Prediction functions
# Option 1: Predict the "maximum likelihood" class y for x, that is, the class maximizing p(x|y).
f_maxlik = function(x) { round(x > mu/2) }

# Option 2: Predict the most probable class y given x, that is, the class maximizing p(y|x).
cutoff_bayes = (1/mu)*(0.5*mu^2 + log((1-p_infected)/p_infected))
f_bayes = function(x) { round(x > cutoff_bayes) }

# Plot prediction functions
lines(x_grid, f_maxlik(x_grid), col=3, lwd=3)
lines(x_grid, f_bayes(x_grid), col=4, lwd=3)


# _____________________________
# Loss function (0-1 loss)
zero_one_loss = function(y_hat,y) { round(y_hat != y) }  # 0-1 loss

# Measure performance (0-1 loss)
print(paste0("Expected 0-1 loss for f_maxlik = ", mean(zero_one_loss(f_maxlik(x), y))))
print(paste0("Expected 0-1 loss for f_bayes =  ", mean(zero_one_loss(f_bayes(x), y))))

# _____________________________
# Loss function (weighted loss)
loss = function(y_hat,y) { 2*((y_hat==1) & (y==0)) + 20*((y_hat==0) & (y==1)) }  # a loss function that more strongly penalizes false negatives

# Measure performance (weighted loss)
print(paste0("Expected loss for f_maxlik = ", mean(loss(f_maxlik(x), y))))
print(paste0("Expected loss for f_bayes =  ", mean(loss(f_bayes(x), y))))

# Option 3: Predict class y with minimum expected loss given x, that is, the class minimizing E(loss(Y_hat, Y) | X=x).
cutoff_minloss = (1/mu)*(mu^2/2 + log((1-p_infected)/p_infected) + log((loss(1,0) - loss(0,0))/(loss(0,1) - loss(1,1))))
f_minloss = function(x) { round(x > cutoff_minloss) }
lines(x_grid, f_minloss(x_grid), col=5, lwd=3)

print(paste0("Expected loss for f_minloss =  ", mean(loss(f_minloss(x), y))))

# Compute loss as a function of the cutoff
loss_grid = sapply(x_grid, function(cutoff) { mean(loss(round(x > cutoff), y)) })  # losses for a range of cutoff points
plot(x_grid, loss_grid, pch=20, col=3, xlab="cutoff", ylab="expected loss")
lines(cutoff_minloss*c(1,1), c(0,2), col=2, lwd=2)  # f_minloss minimizes expected loss


# _____________________________________________________________________________
# Confusion matrix

# Prediction function to use
f = f_minloss

# Confusion matrix
table(f(x), y, dnn=c("y_pred","y_actual"))

# The entries of the confusion matrix are the number of false/true positives/negatives.
TP = sum((f(x)==y) & (f(x)==1))  #  true positives ( "true" =   correct prediction, "positive" = predicted positive)
TP

TN = sum((f(x)==y) & (f(x)==0))  #  true negatives ( "true" =   correct prediction, "negative" = predicted negative)
TN

FP = sum((f(x)!=y) & (f(x)==1))  # false positives ("false" = incorrect prediction, "positive" = predicted positive)
FP

FN = sum((f(x)!=y) & (f(x)==0))  # false negatives ("false" = incorrect prediction, "negative" = predicted negative)
FN

FPR = FP / (FP + TN)  # false positive rate = proportion of actual negatives that were predicted to be positive.
FPR
TPR = TP / (TP + FN)  #  true positive rate = proportion of actual positives that were predicted to be positive.
TPR


# _____________________________________________________________________________
# ROC curve
# (ROC stands for "receiver operating characteristic", but the name is essentially meaningless.)

# Each of f_maxlik, f_bayes, and f_minloss are of the form: round(x > cutoff).
# How does performance vary as a function of cutoff?

o = order(x, decreasing=TRUE)  # order of the x's from largest to smallest
TPR = cumsum(y[o])/sum(y)  # compute the TPR for each of: cutoff=x[o[1]], ..., cutoff=x[o[n]].
FPR = cumsum(1-y[o])/sum(1-y)  # compute the FPR for each of: cutoff=x[o[1]], ..., cutoff=x[o[n]].
plot(FPR, TPR, main="ROC curve", pch=20, col=4)
# The preceding lines are a clever trick for computing ROC curves.



# _____________________________________________________________________________
# Logistic regression

# expit function (a.k.a. logistic function)
expit = function(a) { exp(a)/(1+exp(a)) }
a = seq(-6,6,0.1)
plot(a,expit(a), main="expit function")

# logit function (the inverse of expit)
logit = function(p) { log(p/(1-p)) }
p = seq(0.001,0.999,0.001)
plot(p,logit(p), main="logit function")

# logit and expit are inverses of one another
plot(a,logit(expit(a)))
plot(p,expit(logit(p)))

# Two-dim surface of probabilities under logistic regression model
p = function(x1,x2) { expit(3 - 0.1*x1 - 0.1*x2) }
Pr.class.1 = outer(a,a,p)
persp(a,a,Pr.class.1,theta=300,phi=35)
image(a,a,Pr.class.1)
contour(a,a,Pr.class.1)













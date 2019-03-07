# Lab 5: Penalized regression
# 
# Author: Jeffrey W. Miller
#   (NOTE: Parts 1 and 2 are mostly taken from ISL Chapter 6.)
# Date: March 6, 2019

# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# Part 1: Subset Selection Methods (TEAM EXERCISE)
# (from ISL Chapter 6 Lab 1)

# INSTRUCTIONS:
#   Go through ISL Section 6.5.1, running each step of the code below in R.
#   Write your answers to the questions below.


# _____________________________________________________________________________
# Best Subset Selection

# Load data
#install.packages("ISLR")
library(ISLR)
fix(Hitters)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

# Run best subset selection
#install.packages("leaps")
library(leaps)
regfit.full=regsubsets(Salary~.,Hitters)
summary(regfit.full)
regfit.full=regsubsets(Salary~.,data=Hitters,nvmax=19)
reg.summary=summary(regfit.full)
names(reg.summary)
reg.summary$rsq

# Question: What is the best one-variable model, according to this method?
# Question: What is the best two-variable model, according to this method?

# Plot results
par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(10,reg.summary$cp[10],col="red",cex=2,pch=20)
which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(6,reg.summary$bic[6],col="red",cex=2,pch=20)
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")
coef(regfit.full,6)

# Question: Which subset of variables achieves the best adjusted R-squared score?
# Question: Which subset of variables achieves the best Cp score?
# Question: Which subset of variables achieves the best BIC score?

# _____________________________________________________________________________
# Forward and Backward Stepwise Selection

regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward")
summary(regfit.fwd)
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="backward")
summary(regfit.bwd)
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)

# Try this:
coef(regfit.fwd,6)
# Question: How does this compare to the subset of vars with the best BIC score?

# _____________________________________________________________________________
# Choosing Among Models

# Validation set method (Single train/test split)
set.seed(1)
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE)
test=(!train)
regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)
test.mat=model.matrix(Salary~.,data=Hitters[test,])
val.errors=rep(NA,19)
for(i in 1:19){
   coefi=coef(regfit.best,id=i)
   pred=test.mat[,names(coefi)]%*%coefi
   val.errors[i]=mean((Hitters$Salary[test]-pred)^2)
}
val.errors
which.min(val.errors)
coef(regfit.best,10)

# Function to simplify prediction with regsubsets
predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}

# Re-fit on full data set using 10 variables
regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19)
coef(regfit.best,10)
k=10

# Question: How does this compare to the subset of variables with the best Cp score?
# Question: Does this make sense based on what AIC is "trying" to do?
#   (Cp is equivalent to AIC in this setting.)

# Cross-validation to choose number of variables to include
set.seed(1)
folds=sample(1:k,nrow(Hitters),replace=TRUE)
cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
for(j in 1:k){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
  for(i in 1:19){
    pred=predict(best.fit,Hitters[folds==j,],id=i)
    cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
  }
}
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')

# Re-fit on full data set using k=11
reg.best=regsubsets(Salary~.,data=Hitters, nvmax=19)
coef(reg.best,11)

# Question: Why re-fit the model on the full data set?
# Question: Which would you trust more, using a single train/test split or cross-validation?



# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# Part 2: Ridge Regression and the Lasso (TEAM EXERCISE)
# (from ISL Chapter 6 Lab 2)

# INSTRUCTIONS:
#   Go through ISL Section 6.5.2, running each step of the code below in R.
#   Write your answers the questions below.

x=model.matrix(Salary~.,Hitters)[,-1]
y=Hitters$Salary

# _____________________________________________________________________________
# Ridge Regression

# Fit ridge for a range of lambda values
#install.packages("glmnet")
library(glmnet)
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
dim(coef(ridge.mod))

# Compare coef magnitudes for large vs small lambda
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

# Question: As lambda increases, do the estimated coefficients get larger or smaller?

# The "predict" function can compute coefs for a new lambda value, s
predict(ridge.mod,s=50,type="coefficients")[1:20,]

# Train/test split
set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

# Fit on train set, assess MSE on test set when lambda=4 and lambda=1e10
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])
mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)
ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])
mean((ridge.pred-y.test)^2)

# Question: Which value of lambda (4 or 1e10) gives lower MSE?
# Question: Why is the MSE of lambda=1e10 nearly equal to the MSE of the intercept-only model?

# Compare with MSE of least-squaures
ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train])
mean((ridge.pred-y.test)^2)
lm.fit=lm(y~x, subset=train)
lm.fit
lm.pvalues=summary(lm.fit)$coefficients[,4]
predict(ridge.mod,s=0,exact=T,type="coefficients",x=x[train,],y=y[train])[1:20,]

# Question: Does lm give you the same coefs as ridge with lambda=0?


# Choose lambda using cross-validation with cv.glmnet
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x,y,alpha=0)
ridge.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
ridge.coef

# Question: How does the MSE of ridge using lambda=bestlam compare with lambda=4?
# Question: Are any of the coefs exactly zero?


# _____________________________________________________________________________
# The Lasso

# Fit lasso on the same data set, using a range of lambda values
lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod, xvar="lambda", label=T, xlim=c(-5,6))

# Question: What is the interpretation of this plot?
# Question: For roughly what values of lambda are all the coefs exactly zero?

# Use cross-validation to choose lambda
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

# Assess MSE on the test set when using bestlam
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)

# Question: How does this compare with the MSE of ridge when using the bestlam for ridge?

# Look at the lasso coefs when using bestlam
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]  # nonzero coefs

# Question: Which variables are selected by lasso when using bestlam?

# Question: Which variables are included or not included compared to 
#   using best subset with the same number of variables?
coef(regfit.full,7)  # best subset coefs using 7 variables 

# Compare the pvalues of lm to the variables selected by lasso:
plot(lm.pvalues,lasso.coef!=0,pch=20)
# Would selecting variables using lm.pvalues be roughly the same as lasso?



# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# Part 3: Competition - Predicting age from DNA methylation (TEAM EXERCISE)

# Download lab-5-train.rda from Files / Labs / Lab 5.

# Load the data
# (You will need to change to the directory containing the file, e.g., using setwd.)
load(file="lab-5-train.rda")

# For n=620 subjects, D.train contains the subject's age and 
#   DNA methylation data at p=27578 genomic loci.
D.train[1:5,1:5]  # look at the first few entries
dim(D.train)

# We can extract the ages as y and the methylation data as X.
X = D.train[,-1]  # X[i,] = predictors for subject i
y = D.train[,1]  # y[i] = age of subject i
dim(X)
hist(y)

# The task is to predict age using the methylation data as predictors.
# I have held out a test set with data on 100 additional subjects.
# Your goal is to construct a prediction function to obtain the 
# best MSE on my held-out test set.  You are free to use any 
# regression method (e.g., KNN, least-squares linear regression, 
# ridge, lasso, elastic net, best subset), and any validation 
# method (e.g., cross-validation) to choose model settings.
#
# Define a function called "f_predict" that will make predictions 
# on the test set when called as below.  You cannot use any outside
# information to construct f_predict --- you must use only the
# training data that has been provided.
#
# Each member of the team with the smallest test MSE 
# will get 5 extra credit points on the midterm.
# In the case of a tie, the first team to submit will win.
# *** Team size is limited to 4 people. ***

# Your prediction function
f_predict = function(X.test) { YOUR_CODE_HERE }

# Load test data (when available)
load(file="lab-5-test.rda")
X.test = D.test[,-1]
y.test = D.test[,1]

# Make predictions on the test set
y.test.hat = f_predict(X.test)

# Evaluate test MSE
mean((y.test.hat - y.test)^2)






















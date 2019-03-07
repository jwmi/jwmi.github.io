# Lab 6: Principal components analysis 
# 
# Author: Jeffrey W. Miller
# Date: March 7, 2019
#
# INSTRUCTIONS:
#   Run each step of the code below in R.
#   Write your answers to the questions below.
#   ALL PARTS ARE TEAM EXERCISES.

# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# Part 1: Examples

# _____________________________________________________________________________
# DNA methylation data for epigenetic clock
# Data aggregated by Steve Horvath (former HSPH postdoc)
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41037

load(file="methylation.rda")

# For n=720 subjects, D contains DNA methylation data at p=27578 genomic loci.
dim(D)
p = ncol(D)
n = nrow(D)

# pheno contains information about each sample/subject
pheno = read.table("methylation-phenotype.tsv", header=T, sep="\t", na.strings="NA", row.names=1)
fix(pheno)
summary(pheno)

# Question: Which columns of pheno are self-explanatory?

# High dimensional data like D are hard to visualize in raw form.
# PCA is one way of helping visualize high dimensional data,
# as well as see relationships with sample/subject covariates.

# Run PCA (be patient... this may take a minute)
pc = prcomp(D,scale=T,rank.=2)

# Question: How would you decide whether to use scale=T or F? (There is no "right" answer.)

# Color by used_in_analysis
plot(pc$x[,1:2], col=pheno$used_in_analysis, pch=16)

# A few outliers appear to be present in the used_in_analysis=no group.
# Outliers can obscure the PCA analysis.
# Question: For how many samples is used_in_analysis=no?

# Remove samples not used in original analysis
D = D[pheno$used_in_analysis=="yes",]
pheno = pheno[pheno$used_in_analysis=="yes",]
dim(D)
dim(pheno)

# Rerun PCA
pc = prcomp(D,scale=T,rank.=2)

# Color by age
rbPal = colorRampPalette(c('red','blue'))
color = rbPal(10)[as.numeric(cut(pheno$age, breaks=10))]
plot(pc$x[,1:2], col=color, pch=16)

# Question: Do you see any obvious relationships between PC1,PC2 and age?
# Question: Do you think using PC1 and PC2 to predict age would be very accurate?

# Color by other covariates
plot(pc$x[,1:2], col=pheno$gender, pch=16)
plot(pc$x[,1:2], col=pheno$disease_status, pch=16)
plot(pc$x[,1:2], col=pheno$dataset, pch=16)
plot(pc$x[,1:2], col=pheno$plate, pch=16)
plot(pc$x[,1:2], col=pheno$sentrixposition, pch=16)
plot(pc$x[,1:2], col=pheno$trackingsheet, pch=16)

# Question: Which covariates are clearly correlated with PC1 and/or PC2?

# Dataset, plate, sentrixposition, and trackingsheet are batch variables
# that shouldn't be related to the biology.  

# Question: Based on the plots, do you think that some normalization/preprocessing
# to remove batch effects would be helpful or would it not make much difference?


# _____________________________________________________________________________
# Image compression with PCA
# Adapted from:
#   https://www.r-bloggers.com/image-compression-with-principal-component-analysis/
#   by Aaron Schlegel

# Look at cat.jpg in any image viewer on your computer.
# It is a standard JPEG file.

#install.packages("jpeg")
library(jpeg)
cat = readJPEG("cat.jpg")
ncol(cat)
nrow(cat)

# PCA can be used for constructing lower dimensional "compressed" 
# representations of higher dimensional matrices.

# Extract the red, green, and blue channels as matrices
r = cat[,,1]
g = cat[,,2]
b = cat[,,3]

# Do PCA on each channel (red, green, blue)
cat.r.pca = prcomp(r, center=FALSE)
cat.g.pca = prcomp(g, center=FALSE)
cat.b.pca = prcomp(b, center=FALSE)

# Reconstruct compressed images using increasing numbers of PCs
dir.create("compressed")
rgb.pca = list(cat.r.pca, cat.g.pca, cat.b.pca)
numcomps = seq.int(1, round(nrow(cat)-10), length.out=30)
for (k in numcomps) {
  pca.img = sapply(rgb.pca, function(channel) {
    compressed.img = channel$x[,1:k] %*% t(channel$rotation[,1:k])
  }, simplify = 'array')
  writeJPEG(pca.img,paste('compressed/cat_compressed_',round(k,0),'_components.png',sep=''))
}

# The uncompressed image is represented by 716400 numbers:
#   prod(dim(r))+prod(dim(g))+prod(dim(b)) = 3 * 398 * 600 = 716400.

# Question: If we use one PC for each channel, how many numbers are we using to represent the image?
# Question: What if we use k PCs for each channel?



# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# Part 2: Understanding PCA through code

# _____________________________________________________________________________
# Understanding the eigenvalues and eigenvectors of a covariance matrix

set.seed(1)
library(MASS)  # for generating multivariate normals

# Function to draw a line segment from a to b.
draw_line = function(a,b,col="red",lwd=5,lty=1) { 
  lines(c(a[1],b[1]),c(a[2],b[2]),col=col,lwd=lwd,lty=lty)
}

# Function to plot a bivariate normal with the given parameters
#   mu = mean vector
#   Cov = covariance matrix = ULU'
plot_bvn = function(n,mu,eigval1,eigval2,eigvec1,eigvec2) {
  U = cbind(eigvec1, eigvec2)
  L = diag(c(eigval1,eigval2))
  Cov = U %*% L %*% t(U)
  X = mvrnorm(n, mu, Cov)
  plot(X[,1],X[,2],xlim=c(-6,6),ylim=c(-6,6),cex=0.25)
  draw_line(c(0,0), sqrt(eigval1)*eigvec1)
  draw_line(c(0,0), sqrt(eigval2)*eigvec2)
  return(X)
}

n = 10000  # number of samples to draw
mu = c(0,0)  # mean of bivariate normals to plot

# Standard normal
plot_bvn(n,mu,eigval1=1,eigval2=1,eigvec1=c(1,0),eigvec2=c(0,1))

# The eigenvectors are the directions along which the distribution is aligned.
# The eigenvalues are the variances along these directions.

# Adjust eigenvalues (i.e., variance in direction of eigenvectors)
plot_bvn(n, mu, 2^2, 1, c(1,0), c(0,1))
plot_bvn(n, mu, 2^2, 0.5^2, c(1,0), c(0,1))
plot_bvn(n, mu, 1, 3^2, c(1,0), c(0,1))

# Explain these figures.
# Question: What do the black dots represent?
# Question: What do the red lines represent?
# Question: Why did I write the eigenvalues as x^2 (something squared)?

# Adjust eigenvectors (i.e., directions)
plot_bvn(n, mu, 2^2, 1, c(1,0), c(0,1))
plot_bvn(n, mu, 2^2, 1, c(1,1)/sqrt(2), c(-1,1)/sqrt(2))
# (The eigenvectors must be orthogonal and have length 1.)
plot_bvn(n, mu, 2^2, 1, c(2,1)/sqrt(5), c(-1,2)/sqrt(5))
plot_bvn(n, mu, 2^2, 1, c(3,1)/sqrt(10), c(-1,3)/sqrt(10))

# Play around with the eigenvalues above to see what happens.
# Question: What could you enter to make |X1-X2| small with high probability?

# Adjust both the eigenvalues and eigenvectors
plot_bvn(n, mu, 2^2, 0.5^2, c(3,1)/sqrt(10), c(-1,3)/sqrt(10))
plot_bvn(n, mu, 0.8^2, 0.5^2, c(3,1)/sqrt(10), c(-1,3)/sqrt(10))
plot_bvn(n, mu, 0.8^2, 1.5^2, c(3,1)/sqrt(10), c(-1,3)/sqrt(10))

# We can project each point onto each eigvec:
eigval1 = 2^2
eigval2 = 1
eigvec1 = c(3,1)/sqrt(10)
eigvec2 = c(-1,3)/sqrt(10)
X = plot_bvn(n, mu, eigval1, eigval2, eigvec1, eigvec2)
score1 = X %*% eigvec1
score1[1:5]  # score1[i] = distance from (0,0) to projection of X[i,], along eigvec1
i = 1  # Let's look at point i
draw_line(X[i,], score1[i]*eigvec1, col="blue", lwd=5)
draw_line(c(0,0), score1[i]*eigvec1, col="green", lwd=3, lty=1)

# Re-run the preceding block of code a few times to get a visual
# sense of what the code is doing.

# Question: What are the interpretations of the green and blue lines?


# We can plot the scores (i.e., rotated version of the data, so that
# the eigenvectors are aligned with the axes.)
score2 = X %*% eigvec2
plot(score1,score2, xlim=c(-6,6),ylim=c(-6,6),cex=0.25)
draw_line(c(0,0),sqrt(eigval1)*c(1,0))
draw_line(c(0,0),sqrt(eigval2)*c(0,1))

# Question: Geometrically/intuitively, what is the relationship between 
# the cloud of black points in this plot (i.e., the scores) and 
# the cloud of black points in the previous plot (i.e., the data points). 


# _____________________________________________________________________________
# Covariance method of computing PCA

# In the previous section, we defined the scales and directions
# and generated a collection of data points with those scales along
# those directions.
#
# Conversely, we can take a collection of data points, and
# find the scales and directions (i.e., the sqrt(eigvals) and eigvecs).
# The directions can then be used to compute the scores. This is PCA!

# Do PCA using the covariance method
pca_cov = function(X) {
  Cov_hat = cov(X)  
  eig = eigen(Cov_hat)
  # Note: eigen() returns the eigvecs/vals in sorted order, from largest to smallest eigval.
  directions = eig$vectors
  scales = sqrt(eig$values)
  scores = X %*% directions
  return(list(scores=scores, directions=directions, scales=scales))
}

# Question: Add a comment to each line of pca_cov to explain/justify it.


# Do PCA and plot the PCA directions 
pca = pca_cov(X)
plot(X[,1], X[,2], xlim=c(-6,6),ylim=c(-6,6),cex=0.25)
draw_line(c(0,0),pca$scales[1]*pca$directions[,1])
draw_line(c(0,0),pca$scales[2]*pca$directions[,2])

# Question: Do the red lines here exactly match the red lines in the plot above
#    based on the true scales and directions?
# Question: How do they differ?

# Plot PCA scores
plot(pca$scores[,1],pca$scores[,2], xlim=c(-6,6),ylim=c(-6,6),cex=0.25)

# Compare estimated eigvals/vecs to true eigvals/vecs
eigval1
pca$scales[1]^2
eigval2
pca$scales[2]^2
eigvec1
pca$directions[,1]
eigvec2
pca$directions[,2]

# Each direction is only determined up to a sign flip.


# _____________________________________________________________________________
# Top PCs

# If ncol(X) > 2, then we can use the top few PCs 
# (i.e., scores/directions with largest scale).
# Let's try a slightly more interesting example.
# The data doesn't have to come from a Gaussian.

# 20-dimensional data with two clusters
n = 100
p = 20
X = matrix(rnorm(p*n,0,1),ncol=p)
X[1:(n/2),] = X[1:(n/2),] + 1
plot(X[,1], X[,2], pch=16)  # first two data dimensions
plot(X[,1], X[,2], col=((1:n)>(n/2))+3, pch=16)

# Question: How are the two clusters defined in this example?
# Question: Roughly what do you expect the PC1 direction to be?
# Question: Without the color-coding, could you easily distinguish the two clusters
#   by looking only at pairs of coordinates (e.g., dims 1 and 2)?

# Do PCA
pca = pca_cov(X)
pca$directions[,1:2]  # top two PCA directions
# Plot scores for top two PCA directions
plot(pca$scores[,1],pca$scores[,2], pch=16)
plot(pca$scores[,1],pca$scores[,2], col=((1:n)>(n/2))+3, pch=16)

# Question: Can you easily distinguish the two clusters using PC scores 1 and 2?
# Question: Was your guess regarding the PC1 direction correct?


# If PC1 represents unwanted variation, we can subtract it off:
X_adjusted = X - pca$scores[,1] %*% t(pca$directions[,1])
pca_adj = pca_cov(X_adjusted)
plot(pca_adj$scores[,1],pca_adj$scores[,2], col=((1:n)>(n/2))+3, pch=16)

# Question: What would be an example of unwanted variation? 
#    (The methylation data provided some examples of unwanted variation.)
# Question: The two "clusters" are no longer distinguishable. Why is that desirable
#    if the variation is unwanted?

# Check that we get the same result as prcomp:
pc = prcomp(X)

pca$scales  # we called them "scales"
pc$sdev  # prcomp calls them "sdev"

pca$directions[1:5,1:5]  # we called them "directions"
pc$rotation[1:5,1:5]  # prcomp calls them "rotation"
colSums(pca$directions * pc$rotation)

# Question: What does a 1 or -1 mean here?
# (Note: -1 is fine, since the direction is only determined up to a sign.)

pca$scores[1:5,1:5]  # we called them "scores"
pc$x[1:5,1:5]  # prcomp calls them "x"

# Our scores differ from prcomp's x since prcomp computes centered scores:
colMeans(pc$x)

# Question: Why does this confirm that pc$x is centered?

# We can compute centered scores using:
centered_scores = scale(pca$scores,center=T,scale=F)
centered_scores[1:5,1:5]

# Question: Is centered_scores equal to pc$x or not? If not, in what way do they differ?









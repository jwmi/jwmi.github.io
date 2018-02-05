# Julia code for the article:
#     Fast and accurate approximation of the full conditional for gamma shape parameters. Jeffrey W. Miller, 2018.
#
# Please cite the article if you use this algorithm.  See the end of this file for license info.
# 
# ___ Julia language ___
# This source code is written in the Julia language (https://julialang.org/).  The code was implemented using Julia v0.6.0, so if you use a different version of Julia then tweaks might be needed.  Only the SpecialFunctions package is needed for the algorithm itself, but to make the plots, the following Julia packages are also required: Distributions, PyPlot. These packages can be installed at the Julia command line by running:
#    Pkg.add("SpecialFunctions")
#    Pkg.add("Distributions")
#    ENV["PYTHON"]=""
#    if (Pkg.installed("PyCall")!=nothing); Pkg.build("PyCall"); end
#    Pkg.add("PyPlot")
#    using PyPlot
#
# To execute a given Julia file, say gamma-shape.jl, cd to the appropriate folder, e.g., cd("../folder"), and do:
#    include("gamma-shape.jl")


# __________________________________________________________________________________________________
# Algorithm 1 from the article  (This part is self-contained, and requires only the SpecialFunctions package.)

using SpecialFunctions

# Function to compute parameters A,B for a Gamma(a|A,B) approximation to the full conditional p(a|x,m,a0,b0)
# for the shape parameter a, under the model X_1,...,X_n|a,m i.i.d. ~ Gamma(shape=a, rate=a/m)
# with prior a ~ Gamma(shape=a0, rate=b0).  The approximation is p(a|x,m,a0,b0) \approx Gamma(a | shape=A, rate=B).
function approximate_full_conditional(x,m,a0,b0; tolerance=1e-8, max_iterations=10, verbose=false)
    n = length(x); S = sum(x); R = sum(log.(x + eps(0.)))  # pre-compute the sufficient statistics
    T = S/m - R + n*log(m) - n
    A = a0 + n/2; B = b0 + T  # initialize using Stirling's approximation to the gamma function
    for iteration = 1:max_iterations
        a = A/B
        D = n*(1 - a*trigamma(a))
        A = a0 - a*D
        B = b0 - D - n*(log(a) - digamma(a)) + T
        if verbose; println("a = ",a); end
        if abs(a/(A/B) - 1) < tolerance  # stop if converged
            if verbose
                println("a = $(A/B)   A = $A   B = $B")
                println("Last difference was $(abs(a/(A/B)-1)) after $iteration iterations.")
            end
            return A,B,iteration
        end
        if iteration==max_iterations; warn("Possible non-convergence -- maximum number of iterations reached."); end
    end
    return A,B,max_iterations
end


# __________________________________________________________________________________________________
# Function to compare the true and approximate full conditionals

using Distributions

# Compare the true and approximate full conditionals by computing total variation and KL using numerical integration.
function compare(x,m,A,B,a0,b0,N)
    u = ((1:N)-0.5)/N  # uniform grid
    a = quantile(Gamma(A,1/B),u)  # quantiles of approximate full conditional (Note: Julia uses Gamma(shape,scale).)
    f_approx = pdf.(Gamma(A,1/B),a)  # density of the approximate full conditional at the quantiles
    n,S,R = length(x),sum(x),sum(log.(x+eps(0.)))  # sufficient statistics
    loglik = n*a.*log.(a/m) - n*lgamma.(a) + (a-1)*R - (a/m)*S  # log p(x|a,m)
    f_true_unnormalized = (r = loglik + (a0-1)*log.(a) - b0*a; exp.(r - r[indmin(abs.(a-A/B))]))  # unnormalized density of the true full conditional
    Z = mean(f_true_unnormalized./f_approx)  # numerically compute the normalizing constant of f_true
    f_true = f_true_unnormalized/Z  # density of the true full conditional
    f_ratio = f_true./f_approx  # ratio of true to approximate full conditional density at the quantiles
    
    # compute distances
    d_TV = mean(0.5*abs.(f_ratio - 1))  # total variation distance between true and approximate
    d_KL1 = mean(-log.(f_ratio))  # KL divergence between approximate and true: KL(f_approx||f_true)
    d_KL2 = mean(f_ratio.*log.(f_ratio))  # KL divergence between true and approximate: KL(f_true||f_approx)
    
    # compute cdfs at the quantiles
    F_approx = u
    F_true = cumsum(f_ratio)/N
    
    return f_true,f_approx,F_true,F_approx,a,d_TV,d_KL1,d_KL2
end


# __________________________________________________________________________________________________
# Code to generate the CDFs figure in the article

using PyPlot
drawnow() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
latex(s) = latexstring(replace(s," ","\\,\\,"))

# Settings
srand(0)  # reset the RNG
a0s = [0.01,0.1,1.0]  # prior on the shape: a ~ Gamma(shape=a0, rate=a0/m0)
m0 = 1.0  # prior on the shape: a ~ Gamma(shape=a0, rate=a0/m0)
n_values = [1,10,100]  # sample sizes to use
ratio = 1.0  # m/m_true (ratio of m_true to the value of m to condition on)
a_true = 1.0  # true shape --- the data is x_1,...,x_n i.i.d. ~ Gamma(shape=a_true, rate=a_true/m_true)
m_true = 1.0  # true mean
N = 10000  # number of points for numerical integration (only needed for comparison with the true full conditional)

# Run
for a0 in a0s
    for (i_n,n) in enumerate(n_values)
        println("============== n = $n ==============")
        
        # Run
        x = rand(Gamma(a_true,m_true/a_true),n)  # simulate data (Note: Julia uses Gamma(shape,scale).)
        m = ratio*m_true  # value of m to use (in a sampler, this would be the current value of m)
        A,B,n_iterations = approximate_full_conditional(x,m,a0,a0/m0; verbose=true)  # approximate the full conditional by Gamma(shape=A, rate=B).
        f_true,f_approx,F_true,F_approx,a,d_TV,d_KL1,d_KL2 = compare(x,m,A,B,a0,a0/m0,N)  # compare the true and approximate full conditionals

        # Plot results    
        figure(1,figsize=(5.5,3.2)); clf()
        subplots_adjust(bottom=0.2)
        title(latex("n=$n, a_0=$a0"),fontsize=16)
        plot(a,F_approx,"r",lw=1,label="approx")
        plot(a,F_true,"b--",lw=2,label="true")
        xlabel(L"a",fontsize=14)
        ylabel(L"\mathrm{CDF}(a|\mu,x_{1\!:\!n})",fontsize=14)
        ylim(0,1)
        legend(loc="lower right",fontsize=14)
        drawnow()
        savefig("cdf-a0=$a0-m0=$m0-n=$n-ratio=$ratio-a=$a_true-m=$m_true.png",dpi=150)
        close()

        println("Total variation distance = ",d_TV)
        println("KL(f_approx||f_true) = ",d_KL1)
        println("KL(f_true||f_approx) = ",d_KL2)
    end
end


# __________________________________________________________________________________________________
# Code to generate the rest of the figures and the table of iterations required

# Settings
srand(0)
a0s = [0.01,0.1,1.0]  # prior on the shape: a ~ Gamma(shape=a0, rate=a0/m0)
m0 = 1.0  # prior on the shape: a ~ Gamma(shape=a0, rate=a0/m0)
n_values = [1,10,100]  # sample sizes to use
ratios = [0.5,1,2]  # m/m_true (ratio of m_true to the value of m used for conditioning)
a_trues = logspace(6,-6,13)  # range of true shapes to use
m_trues = logspace(-6,6,13)  # range of true means to use
n_reps = 5  # number of times to repeat each simulation
N = 10000  # number of points for numerical integration (only needed for comparison with the true full conditional)

# Record-keeping
na0s,nns,nrs,nas,nms = length(a0s),length(n_values),length(ratios),length(a_trues),length(m_trues)
d_TVs = zeros(na0s,nns,nrs,nas,nms,n_reps)
d_KL1s = zeros(na0s,nns,nrs,nas,nms,n_reps)
d_KL2s = zeros(na0s,nns,nrs,nas,nms,n_reps)
iterations = zeros(Int,na0s,nns,nrs,nas,nms,n_reps)

# Run
for (i_a0,a0) in enumerate(a0s)
    for (i_n,n) in enumerate(n_values)
        for (i_r,ratio) in enumerate(ratios)
            println("a0 = $a0  n = $n  ratio = $ratio")
            for (i_a,a_true) in enumerate(a_trues)
                for (i_m,m_true) in enumerate(m_trues)
                    for rep = 1:n_reps
                        x = rand(Gamma(a_true,m_true/a_true),n)  # simulate data (Note: Julia uses Gamma(shape,scale).)
                        m = m_true*ratio  # value of m to use (in a sampler, this would be the current value of m)
                        A,B,n_iterations = approximate_full_conditional(x,m,a0,a0/m0)  # approximate the full conditional by Gamma(shape=A, rate=B).
                        f_true,f_approx,F_true,F_approx,a,d_TV,d_KL1,d_KL2 = compare(x,m,A,B,a0,a0/m0,N)  # compare the true and approximate full conditionals
                        d_TVs[i_a0,i_n,i_r,i_a,i_m,rep] = d_TV
                        d_KL1s[i_a0,i_n,i_r,i_a,i_m,rep] = d_KL1
                        d_KL2s[i_a0,i_n,i_r,i_a,i_m,rep] = d_KL2
                        iterations[i_a0,i_n,i_r,i_a,i_m,rep] = n_iterations
                    end
                end
            end
        end
    end
end

# Number of iterations required
for (i_a0,a0) in enumerate(a0s)
    I = iterations[i_a0,:,:,:,:,:]
    println("\nHistogram of iterations required (a0 = $a0):")
    for i = 1:maximum(I)
        println("$i iterations were required in $(sum(I.==i)) of the runs")
    end
end

# Heatmaps of case-by-case distances
rstr = ["0.5","","2"]  # strings corresponding to ratios, for plot titles
for (distances,dname) in [(d_TVs,"d_TV"),(d_KL1s,"d_KL1"),(d_KL2s,"d_KL2")]
    for (i_a0,a0) in enumerate(a0s)
        for (i_n,n) in enumerate(n_values)
            vmax = 1.4*maximum(mean(distances,6)[i_a0,i_n,:,:,:,:])
            for (i_r,ratio) in enumerate(ratios)
                figure(1, figsize=(5.5,5.5)); clf()
                subplots_adjust(bottom=0.2)
                title(latex("n=$n, \\mu=$(rstr[i_r])\\\mu_\\mathrm{true}"),fontsize=16)
                D = squeeze(mean(distances[i_a0,i_n,i_r,:,:,:],3),3)
                # imshow(D; vmax=vmax, cmap=PyPlot.cm_get_cmap("binary"))
                imshow(D; vmax=vmax, cmap=PyPlot.cm_get_cmap("YlGnBu"))
                ylabel(L"\log_{10}(a_\mathrm{true})",fontsize=14)
                xlabel(L"\log_{10}(\mu_\mathrm{true})",fontsize=14)
                yticks(0:nas-1,round.(Int,log10.(a_trues)),fontsize=12)
                xticks(0:nms-1,round.(Int,log10.(m_trues)),fontsize=12)
                colorbar()
                drawnow()
                savefig("$dname-a0=$a0-n=$n-ratio=$ratio.png",dpi=150)
                close()
            end
        end
    end
end

# Plot worst-case distances
for (distances,dname,dstr) in [(d_TVs,"d_TV",L"d_\mathrm{TV}(f,g)"),(d_KL1s,"d_KL1",L"d_\mathrm{KL}(g,f)"),(d_KL2s,"d_KL2",L"d_\mathrm{KL}(f,g)")]
    figure(1, figsize=(5.5,3.2)); clf()
    subplots_adjust(bottom=0.2)
    title(dstr,fontsize=16)
    mark = ["o","^","s"]
    for (i_a0,a0) in enumerate(a0s)
        D = squeeze(mean(distances[i_a0,:,:,:,:,:],5),5)
        d_max = maximum(D,[2,3,4])[:]
        loglog(n_values,d_max,mark[i_a0]*"-",lw=2,label=latex("a_0 = $a0"))
    end
    legend(loc="upper right",fontsize=14)
    ylabel("max discrepancy",fontsize=14)
    xlabel(L"n",fontsize=14)
    yl = ylim()
    # yticks(0:.01:.1)
    ylim(yl[1],0.1)
    drawnow()
    savefig("worst-case-$dname.png",dpi=150)
    close()
end



nothing



# __________________________________________________________________________________________________
# LICENSE

# gamma-shape.jl is licensed under the MIT "Expat" License:
# 
# Copyright (c) 2018: Jeffrey W. Miller.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


























using PyPlot
using Delimited Files

x = readdlm("Patterson.dat")

I,J = size(x)

imshow(x; cmap="BrBG",vmin=-0.5,vmax=1.25)
xticks(0:J-1,1:J; fontsize=7)
yticks(0:I-1,1:I; fontsize=7)
xlabel("Habitat")
ylabel("Species")

savefig("cooccurrence.png";dpi=200)


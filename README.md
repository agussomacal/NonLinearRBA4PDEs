# Non Linear Compressive Reduced Basis Approximation for PDE’s

This repository has the implementations of the article: **Non Linear Compressive Reduced Basis Approximation for PDE’s**

Abstract: Linear model reduction techniques design *offline* low-dimensional subspaces that are tailored to the approximation of solutions to a parameterized partial differential equation, for the purpose of fast *online* numerical simulations. These methods, such as the POD or Reduced Basis (RB) methods, are very effective when the family of solutions has fast-decaying Karhunen-Lo\`eve eigenvalues or Kolmogorov widths, reflecting the approximability by finite-dimensional linear spaces. On the other hand, they become ineffective when these quantities have a slow decay,
in particular for families of solutions to hyperbolic transport equations with parameter-dependent shock positions. The objective of this work is to explore the ability of nonlinear 
model reduction to circumvent this particular situation. To this end, we first describe particular notions of non-linear widths that have a substantially faster decay for the aforementioned families. 
Then, we discuss a systematic approach for achieving better performance via a non-linear reconstruction
from the first coordinates of a linear reduced model approximation, thus allowing us to stay in the same "classical" framework of projection-based model reduction. We analyze the approach and report on its performance for a simple and yet instructive univariate test case.

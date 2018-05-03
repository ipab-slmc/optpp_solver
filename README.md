# EXOTica solvers based on OPT++

This package depends on:
 - exotica
 - optpp_catkin

To run the examples you will also need:
 - exotica_python
 - exotica_examples

### Return codes

```
-15 = error returned by MPI_Allreduce
-14 = error returned by MPI_Initialized
-10 = memory error
-4 = Maximum number of iterations or fevals
-1 = Step does not satisfy sufficient decrease condition
>0 = converged (checkConvgd)

------
MCSRCH
 *       info =-1 improper input parameters. 
 *       info = 1  the sufficient decrease condition and the 
 *                 directional derivative condition hold. 
 *       info =-2  relative width of the interval of uncertainty 
 *                 is at most xtol. 
 *       info =-4  the step is at the lower bound stpmin. 
 *       info =-5  the step is at the upper bound stpmax. 
 *       info =-6  rounding errors prevent further progress. 
 *                 there may not be a step which satisfies the 
 *                 sufficient decrease and curvature conditions. 
 *                 tolerances may be too small. 
```

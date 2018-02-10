# Lasso Quantile Regression

This code provides a Python implementation of the  the lasso quantile regression algorithm of Li and Zhu (2008). 
The paper is available at http://dx.doi.org/10.1198/106186008X289155 ([here](http://dept.stat.lsa.umich.edu/~jizhu/pubs/Li-JCGS08.pdf) is the working paper version).

The major difference to alternatives such as hqreg (see below) is that this algorithm directly solves the constrained regression problem and not the Lagrangian formulation.
This is for instance convenient in forecast combination problems due to the similarity of the lasso and convex quantile regression.

## Example

In the repository, you can find the daily log returns of the IBM stock and the corresponding 1% VaR forecasts stemming from a variety of risk models.

The trace plot below is the result of a lasso quantile regression of the returns on the standalone forecasts.

![Alt text](/output/trace_plot.png)

## Alternatives

The [quantreg](https://cran.r-project.org/web/packages/quantreg/index.html) package for R can estimate the lasso quantile regression for single penalization levels.

The [hqreg](https://cran.r-project.org/web/packages/hqreg/index.html) package for R implements the algorithm of [Yi and Huang](https://arxiv.org/abs/1509.02957), which estimates the whole path of elastic net penalized quantile regression.

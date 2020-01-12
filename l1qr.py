import logging
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from numpy.linalg import LinAlgError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class L1QR:
    def __init__(self, y: pd.Series, x: pd.DataFrame, alpha: float) -> None:
        """Python implementation of the L1 norm QR algorithm of
        Li and Zhu (2008): L1-Norm Quantile Regression, http://dx.doi.org/10.1198/106186008X289155

        Args:
            y: Vector of response data
            x: Matrix of covariates
            alpha: Quantile of interest
        """
        self.x = x.to_numpy()
        self.y = y.to_numpy()
        self.var_names = x.columns
        self.alpha = alpha

        # set by fit()
        self.beta0: Optional[np.array] = None
        self.beta: Optional[np.array] = None
        self.s: Optional[np.array] = None
        self.b0: Optional[pd.Series] = None
        self.b: Optional[pd.DataFrame] = None

    def fit(self, s_max: float = np.inf) -> None:
        """Estimate the model.

        Args:
            s_max: Stop the algorithm prematurely when the L1 norm of the slope coefficients reaches s_max
        """
        n, k = self.x.shape
        if self.y.size != n:
            raise Exception('y and x have different number of rows!')
        logger.info(f'Initialization lasso quantile regression for n={n}, k={k}, and alpha={self.alpha}')

        xc = np.hstack((np.ones((n, 1)), self.x))  # Store x a second time with intercept
        eps1 = 10 ** -10                           # Some low value
        eps2 = 10 ** -10                           # Convergence criterion
        max_steps = n * np.min((k, n - 1))         # Maximum number of steps for the algorithm
        ind_n = np.arange(n)                       # Index of the observations
        ind_k = np.arange(k)                       # Index of the variables
        beta0 = np.zeros((max_steps + 1, 1))       # Stores the estimates of the constant term
        beta = np.zeros((max_steps + 1, k))        # Stores the estimates of the slope parameters
        s = np.zeros(max_steps + 1)                # Stores the penalty parameter

        y_can_be_ordered_strictly = np.unique(self.y).size != n
        if y_can_be_ordered_strictly:
            logger.info('Adding noise to y because y contains duplicate values')
            self.y += np.random.normal(loc=0, scale=10 ** -5, size=self.y.size)

        logger.info('Finding initial solution')

        # There are actually two cases, first if n*tau is integer, second if tau*n is non-integer.
        # Here I assume that in the latter case all the weight is on the first component (see section 2.2)
        ini_beta0 = np.sort(self.y)[int(np.floor(self.alpha * n))]  # Initial beta0 (see 2.2.1)
        ini_beta = np.zeros(k)                                      # Initial beta (see 2.2.1)

        ind_e = np.array(int(np.argwhere(self.y == ini_beta0)))  # Index of the first point in the elbow
        ind_l = ind_n[self.y < self.y[ind_e]]                    # All points that are left of the elbow
        ind_r = ind_n[self.y > self.y[ind_e]]                    # All points that are right of the elbow
        residual = self.y - ini_beta0                            # Initial residuals

        # Add the first variable to the active set
        inactive = ind_k                                # All variables not in V
        tmp_e, tmp_l, tmp_r = ind_e, ind_l, ind_r       # Create a copy of the index sets
        lambda_var = np.zeros((2, inactive.size))       # First row: sign=1, second row: sign=-1
        lambda_var[lambda_var == 0] = -np.inf           # Initially set to -inf (want to maximize lambda)
        b = np.array([0, 1])                            # The 1_0 vector (see p. 171 bottom)
        nu_var = np.zeros((2, inactive.size, b.size))   # 3d array: nu for sign=1 in first dimension, sign=-1 in second

        for j_idx, j_star in enumerate(inactive):
            x_v = xc[:, np.append(0, j_star + 1)]

            # Sign of the next variable to include may be either positive or negative
            for sign in (1, -1):
                index = np.where(sign == 1, 0, 1)  # Index in nu_var and lambda_var

                # Combination of (2.10) and (2.11)
                x0 = np.vstack((np.hstack((1, np.mat(self.x)[tmp_e, j_star])), np.hstack((0, sign))))

                try:  # Check if x0 has full rank
                    nu_tmp = np.linalg.solve(x0, b)  # Solve system (p. 171 bottom)
                    nu_var[index, j_idx, :] = nu_tmp

                    # Store sets that are used to compute -lambda* (p. 172)
                    x_l = x_v.take(tmp_l, axis=0, mode='clip')
                    x_r = x_v.take(tmp_r, axis=0, mode='clip')

                    # Save lambda achieved by the current variable. If sign of last entry != sign then leave at -inf.
                    if np.sign(nu_tmp[-1]) == sign:
                        lambda_var[index, j_idx] = -((1 - self.alpha) * np.dot(x_l, nu_tmp).sum() -
                                                     self.alpha * np.dot(x_r, nu_tmp).sum())
                except LinAlgError:
                    logger.debug(f'sign: {sign}')

        # Select the nu corresponding to the maximum lambda and store the maximum lambda
        nu_var = nu_var[lambda_var.argmax(axis=0), np.arange(inactive.size), :]
        lambda_var = lambda_var.max(axis=0)

        # Store the active variable
        ind_v = inactive[lambda_var.argmax()]

        # Store initial nu0 and nu
        nu0 = nu_var[ind_v, 0]
        nu = nu_var[ind_v, 1:]

        beta0[0] = ini_beta0
        beta[0] = ini_beta
        logger.debug(f'Initial beta0: {ini_beta0}')
        logger.debug(f'Initial beta: {ini_beta}')

        # Main loop
        logger.info('Entering main loop')
        drop = False
        idx = 0
        while idx < max_steps:
            logger.debug(f'Index: {idx}')
            idx += 1

            # Calculate how far we need to move (the minimum distance between points and elbow)
            if np.atleast_1d(nu).size == 1:  # Make sure scalar array is converted to float, causes problems with np.dot
                nu = np.float(nu)

            # (2.14), nu0 + x'*nu where x is without i in elbow
            gam = nu0 + np.dot(self.x.take(ind_n[np.in1d(ind_n, ind_e, invert=True)], axis=0).take(ind_v, axis=1), nu)
            gam = np.ravel(gam)  # Flatten the array
            delta1 = np.delete(residual, ind_e, 0) / gam  # This is s - s_l in (2.14)

            # Check whether all points are in the elbow or if we still need to move on
            if np.sum(delta1 <= eps2) == delta1.size:
                delta = np.inf
            else:
                delta = delta1[delta1 > eps1].min()

            # Test if we need to remove some variable j from the active set
            if idx > 1:
                delta2 = np.array(-beta[idx - 1, ind_v] / nu)

                if np.sum(delta2 <= eps2) == delta2.size:
                    tmpz_remove = np.inf
                else:
                    tmpz_remove = delta2[delta2 > eps1].min()

                if tmpz_remove < delta:
                    drop = True
                    delta = tmpz_remove
                else:
                    drop = False

            # Check if we need to continue or if we are done
            if delta == np.inf:
                logger.info(f'Finished, delta = inf')
                break

            # Update the shrinkage parameter
            s[idx] = s[idx - 1] + delta

            # Prepare the next steps depending if we drop a variable or not
            if drop:
                tmp_delta = delta2[delta2 > eps1]  # All deltas larger than eps2
                tmp_ind = ind_v[delta2 > eps1]  # All V larger than eps2
                j1 = tmp_ind[tmp_delta.argmin()]  # The index of the variable to kick out
            else:
                # Find the i that will hit the elbow next
                tmp_ind = np.delete(ind_n, ind_e)[
                    delta1 > eps2]  # Remove Elbow from observations and keep non-zero elements
                tmp_delta = delta1[delta1 > eps2]  # All deltas that are non-zero
                i_star = tmp_ind[tmp_delta.argmin()]

            # Update beta
            beta0[idx] = beta0[idx - 1] + delta * nu0
            beta[idx] = beta[idx - 1]
            beta[idx, ind_v] = beta[idx - 1, ind_v] + delta * nu

            if s[idx] > s_max:
                logger.info(f's = {s[idx]:.2f} is large enough')
                break

            # Reduce residuals not in the elbow by delta*gam
            residual[np.in1d(ind_n, ind_e, invert=True)] -= delta * gam

            # Check if there are points in either L or R if we do not drop
            if (ind_l.size + ind_r.size == 1) & (not drop):
                logger.info('No point in L or R')
                break

            # Add a variable to the active set
            # Test if all variables are included. If yes, set lambda_var to -inf and continue with next step
            if ind_v.size == k:
                lambda_var = np.zeros((2, inactive.size))
                lambda_var[lambda_var == 0] = -np.inf
            else:
                inactive = ind_k[np.in1d(ind_k, ind_v, invert=True)]  # All variables not in V
                tmp_e, tmp_l, tmp_r = ind_e, ind_l, ind_r  # Create a copy of the index sets

                if drop:
                    ind_v = ind_v[ind_v != j1]  # Remove the detected variable from V
                else:
                    # Add i_star to the Elbow and remove it from either Left or Right
                    # (we know that i_star hits the elbow)
                    tmp_e = np.append(tmp_e, i_star)
                    tmp_l = tmp_l[tmp_l != i_star]
                    tmp_r = tmp_r[tmp_r != i_star]

                lambda_var = np.zeros((2, inactive.size))  # First row: sign=1, second row: sign=-1
                lambda_var[lambda_var == 0] = -np.inf  # Initially set to -inf (want to maximize lambda)
                nu_var = np.zeros((2, inactive.size, 1 + ind_v.size + 1))  # Store nus in 3d array
                b = np.array([0] * (ind_v.size + 1) + [1])  # The 1_0 vector (see p. 171 bottom)

                for j_idx in range(inactive.size):
                    j_star = inactive[j_idx]  # Select variable j as candidate for the next active variable

                    # Select all columns of x that are in ind_v and additionally j_star.
                    # Transposition improves performance as Python stores array in row-major order
                    x_v = xc.T.take(np.append(0, np.append(ind_v, j_star) + 1), axis=0, mode='clip').T

                    # Combination of (2.10) and (2.11)
                    x0 = np.vstack((np.hstack((np.ones((tmp_e.size, 1)),
                                               self.x[tmp_e][:, ind_v].reshape((tmp_e.size, -1)),
                                               self.x[tmp_e, j_star].reshape((tmp_e.size, -1)))),
                                    np.hstack(
                                        (0, np.sign(beta[idx, ind_v]), np.nan))))  # nan is a placeholder for sign

                    # Sign of the next variable to include may be either positive or negative
                    for sign in (1, -1):
                        index = np.where(sign == 1, 0, 1)  # Index in nu_var and lambda_var
                        x0[-1, -1] = sign  # Change sign in the x0 matrix

                        try:
                            nu_tmp = np.linalg.solve(x0, b)  # Solve system (p. 171 bottom)

                            # If sign of last entry != sign then leave at -inf.
                            if np.sign(nu_tmp[-1]) == sign:
                                nu_var[index, j_idx, :] = nu_tmp
                                # Store sets that are used to compute -lambda* (p. 172))
                                x_l = x_v.take(tmp_l, axis=0, mode='clip')
                                x_r = x_v.take(tmp_r, axis=0, mode='clip')
                                lambda_var[index, j_idx] = -((1 - self.alpha) * np.dot(x_l, nu_tmp).sum() -
                                                             self.alpha * np.dot(x_r, nu_tmp).sum())
                        except LinAlgError:
                            pass

                # Select the maximum of each column
                nu_var = nu_var[lambda_var.argmax(axis=0), np.arange(inactive.size), :]
                lambda_var = lambda_var.max(axis=0)

            # Remove an observation from the elbow
            lambda_obs = np.zeros(tmp_e.size)
            lambda_obs[lambda_obs == 0] = -np.inf
            nu_obs = np.zeros((1 + ind_v.size, tmp_e.size))
            left_obs = np.zeros(tmp_e.size)  # 1 if if we shifted observation to the left
            b = np.array([0] * ind_v.size + [1])

            # Store the L and the R observations of x
            x_v = xc.T.take(np.append(0, ind_v + 1), axis=0, mode='clip').T
            x_r = x_v.take(tmp_r, axis=0, mode='clip')
            x_l = x_v.take(tmp_l, axis=0, mode='clip')

            # Combination of (2.10) and (2.11), here without an additional variable j
            x0_all = np.vstack((np.hstack((np.ones((tmp_e.size, 1)), self.x[tmp_e][:, ind_v].reshape((tmp_e.size, -1)))),
                                np.hstack((0, np.sign(beta[idx, ind_v])))))

            for i in range(tmp_e.size):
                x0 = np.delete(x0_all, i, 0)  # Delete the ith observation
                try:
                    nu_tmp = np.linalg.solve(x0, b)  # Solve system (p. 171 bottom)
                    nu_obs[:, i] = nu_tmp
                    # Save lambda achieved by removing observation i
                    lambda_obs[i] = -((1 - self.alpha) * np.dot(x_l, nu_tmp).sum() -
                                      self.alpha * np.dot(x_r, nu_tmp).sum())

                    # Test if we shift i to the left or to the right
                    tmpyf = np.dot(np.append(1, self.x[tmp_e[i], ind_v]), nu_obs[:, i])
                    if tmpyf > 0:  # To the left
                        left_obs[i] = 1
                        lambda_obs[i] += -(1 - self.alpha) * tmpyf
                    else:  # To the right
                        lambda_obs[i] += self.alpha * tmpyf
                except LinAlgError:
                    pass

            # Compare the effects of adding one variable to V and removing one observation from E
            lam_var = lambda_var.max()  # Maximum lambda from adding a variable
            lam_obs = lambda_obs.max()  # Maximum lambda from removing an observation from E

            if (lam_var > lam_obs) & (lam_var > 0):  # Add variable to V
                lam = lam_var

                ind_v = np.append(ind_v, inactive[lambda_var.argmax()])

                if not drop:  # Add i_star to the elbow
                    ind_e = np.append(ind_e, i_star)
                    ind_l = ind_l[ind_l != i_star]
                    ind_r = ind_r[ind_r != i_star]

                # Store nu0 and nu
                nu0 = nu_var[lambda_var.argmax(), 0]
                nu = nu_var[lambda_var.argmax(), 1:]

            elif (lam_obs > lam_var) & (lam_obs > 0):  # Remove observation from E
                lam = lam_obs

                # i_star remains in E, no change in V
                if not drop:
                    ind_l = ind_l[ind_l != i_star]
                    ind_r = ind_r[ind_r != i_star]

                # Find a new i_star
                i_star = tmp_e[lambda_obs.argmax()]
                nu0 = nu_obs[0, lambda_obs.argmax()]
                nu = nu_obs[1:, lambda_obs.argmax()]

                # Test if we need to add i_star to L or R
                if left_obs[lambda_obs.argmax()] == 1:
                    ind_l = np.append(ind_l, i_star)
                else:
                    ind_r = np.append(ind_r, i_star)

                # Remove i_star from E
                ind_e = tmp_e[tmp_e != i_star]
            else:
                logger.info('No further descent')
                break

            # Check if descent is too small
            if np.abs(lam) < eps2:
                logger.info('Descent is small enough')
                break

            drop = False

        logger.info('Algorithm terminated')

        # Save the results
        self.beta0 = beta0[:idx][:, 0]
        self.beta = beta[:idx]
        self.s = s[:idx]
        self.var_names = self.var_names

        # Save the interpolated estimates
        s = np.linspace(self.s[0], self.s[-1], 1000)               # Interpolate shrinkage values
        b0 = pd.Series(np.interp(s, self.s, self.beta0), index=s)  # Extract and interpolate intercept
        b = pd.DataFrame(np.apply_along_axis(lambda w: np.interp(s, self.s, w), 0, self.beta),
                         index=s, columns=self.var_names)          # Extract and interpolate slope
        self.b0 = b0
        self.b = b

    def plot_trace(self, file_name=None, size=(8, 6)):
        """Plot the trace of the estimated coefficients

        Args:
            file_name: If None, the figure will be displayed
            size: (width, height) in inches

        """
        plt.ioff()

        b0, b = self.b0, self.b.copy()
        b.columns = [str(i + 1) + ' - ' + str(col) for i, col in enumerate(b.columns)]

        sns.set(style='white', rc={'font.family': 'serif'})

        fig = plt.figure(figsize=size)
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0, ax1 = plt.subplot(gs[0]), plt.subplot(gs[1])

        ax0.set_title("Trace Plot of the Lasso Quantile Regression")

        # Plot slopes
        b.plot(ax=ax0, legend=False, grid=False, linewidth=1.5, colormap='nipy_spectral')
        ax0.legend(loc='upper left', ncol=int(np.ceil(b.shape[1] / 6)),
                   framealpha=0.5, fancybox=True, labelspacing=0.1, columnspacing=0.5, fontsize=8)
        ax0.set_ylabel(r'Estimated Coefficients')

        # Plot intercept
        b0.plot(ax=ax1, legend=False, grid=False, style='gray')
        ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax1.set_ylabel(r'Intercept')
        ax1.set_xlabel('Sum of the Absolute Slope Coefficients')

        # Add labels to right side
        for i in range(b.shape[1]):
            ax0.annotate(i + 1, xy=(ax0.get_xlim()[1], b.iloc[-1][i]), va='center', ha='left', size=7,
                         textcoords='offset points', xytext=(2, 0))

        # Tight layout and save
        sns.despine()
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name,  bbox_inches='tight', pad_inches=0.02)
        plt.close('all')

    def predict(self, x: pd.DataFrame, s: Optional[float] = None) -> pd.Series:
        """Make a prediction.

        Args:
            x: A DataFrame with values to be predicted. The dimensions need to match
               the data used for training the model.
            s: The penalty parameter for which a prediction is returned.

        Returns:
            A Series with predictions for x.
        """
        if s < min(self.s) or s > max(self.s):
            raise ValueError(f's = {s} should be between '
                             f'{min(self.s):.2f} and {max(self.s):.2f}.')

        if set(self.b.columns) != set(x.columns):
            raise ValueError('The columns of b and x do not match!')

        x = x[self.b.columns]
        prediction = self.b0.to_numpy() + self.b.to_numpy().dot(x.to_numpy().T).T

        s0 = np.linspace(self.s[0], self.s[-1], 1000)
        p = pd.Series(np.apply_along_axis(lambda w: np.interp(s, s0, w), 1, prediction),
                      index=x.index)
        return p


if __name__ == "__main__":

    # Test data: the daily log returns of the IBM stock and the 1% VaR forecasts stemming from a variety of risk models.
    Y = pd.read_csv("input/returns.txt", index_col=0).squeeze()
    X = pd.read_csv("input/quantile_predicitons.txt", index_col=0)

    # Split data into an in-sample and an out-sample-part
    share = 0.9
    n = len(Y)
    idx_in = range(int(np.floor(n * share)))
    idx_out = range(int(np.floor(n * share)), n)

    Y_in, X_in = Y.iloc[idx_in], X.iloc[idx_in]
    Y_out, X_out = Y.iloc[idx_out], X.iloc[idx_out]

    # Estimate the lasso quantile regression
    mdl = L1QR(y=Y_in, x=X_in, alpha=0.01)
    mdl.fit(s_max=3)

    # Plot the trace
    mdl.plot_trace(file_name="output/trace_plot.png", size=(10, 6))

    # Make prediction
    pred = mdl.predict(x=X_out, s=1)

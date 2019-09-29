import logging
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from numpy.linalg import LinAlgError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class IndexSets:
    n: np.array
    k: np.array
    elbow: np.array
    left: np.array
    right: np.array
    inactive: np.array
    active: np.array
    tmp_elbow: np.array
    tmp_left: np.array
    tmp_right: np.array


@dataclass
class Coefficients:
    xc: np.array
    initial_beta0: np.array
    initial_beta: np.array
    beta0: np.array
    beta: np.array
    residual: np.array
    s: np.array
    lambda_var: np.array
    nu_var: np.array
    b: np.array
    nu0: float
    nu: np.array


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

    def fit(self, s_max: float = np.inf, eps1: float = 10 ** -10, eps2: float = 10 ** -10) -> None:
        """Estimate the model.

        Args:
            s_max: Stop the algorithm prematurely when the L1 norm of the slope coefficients reaches s_max
            eps1: Some low value
            eps2: Convergence criterion
        """
        n, k = self.x.shape
        if self.y.size != n:
            raise Exception('y and x have different number of rows!')
        logger.info(f'Initialization lasso quantile regression for n={n}, k={k}, and alpha={self.alpha}')

        max_steps_algo = n * np.min((k, n - 1))
        self._check_if_y_can_be_ordered(n)
        _initial_beta0 = self._get_initial_beta0(n)

        coef = Coefficients(
            xc=np.hstack((np.ones((n, 1)), self.x)),
            initial_beta0=_initial_beta0,
            initial_beta=np.zeros(k),
            beta0=np.vstack((_initial_beta0, np.zeros((max_steps_algo, 1)))),
            beta=np.zeros((max_steps_algo + 1, k)),
            residual=self.y - _initial_beta0,
            s=np.zeros(max_steps_algo + 1),
            lambda_var=np.full((2, k), -np.inf),   # First row: sign=1, second row: sign=-1
            b=np.array([0, 1]),                    # The 1_0 vector (see p. 171 bottom)
            nu_var=np.zeros((2, k, 2)),            # 3d array: nu for sign=1 in first dimension, sign=-1 in second
            nu0=0,
            nu=np.array([])
        )
        indx = IndexSets(
            n=np.arange(n),
            k=np.arange(k),
            elbow=np.array(int(np.argwhere(self.y == coef.initial_beta0))),
            tmp_elbow=np.array(int(np.argwhere(self.y == coef.initial_beta0))),
            left=np.arange(n)[self.y < coef.initial_beta0],
            tmp_left=np.arange(n)[self.y < coef.initial_beta0],
            right=np.arange(n)[self.y > coef.initial_beta0],
            tmp_right=np.arange(n)[self.y > coef.initial_beta0],
            inactive=np.arange(k),
            active=np.array([])
        )

        indx, coef = self._assign_first_variable(coef, indx)

        # Main loop
        logger.info('Entering main loop')
        drop = False
        idx = 0
        while idx < max_steps_algo:
            logger.debug(f'Index: {idx}')
            idx += 1

            # Calculate how far we need to move (the minimum distance between points and elbow)
            if np.atleast_1d(coef.nu).size == 1:  # Make sure scalar array is converted to float, causes problems with np.dot
                coef.nu = np.float(coef.nu)

            # (2.14), nu0 + x'*nu where x is without i in elbow
            gam = coef.nu0 + np.dot(self.x.take(indx.n[np.in1d(indx.n, indx.elbow, invert=True)], axis=0) \
                                    .take(indx.active, axis=1), coef.nu)
            gam = np.ravel(gam)  # Flatten the array
            delta1 = np.delete(coef.residual, indx.elbow, 0) / gam  # This is s - s_l in (2.14)

            # Check whether all points are in the elbow or if we still need to move on
            if np.sum(delta1 <= eps2) == delta1.size:
                delta = np.inf
            else:
                delta = delta1[delta1 > eps1].min()

            # Test if we need to remove some variable j from the active set
            if idx > 1:
                delta2 = np.array(-coef.beta[idx - 1, indx.active] / coef.nu)

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
            logger.debug(f'Updating s = {coef.s[idx - 1]:.2f} by delta = {delta:.2f}')
            coef.s[idx] = coef.s[idx - 1] + delta

            # Prepare the next steps depending if we drop a variable or not
            if drop:
                tmp_delta = delta2[delta2 > eps1]  # All deltas larger than eps2
                tmp_ind = indx.active[delta2 > eps1]  # All V larger than eps2
                j1 = tmp_ind[tmp_delta.argmin()]  # The index of the variable to kick out
            else:
                # Find the i that will hit the elbow next
                tmp_ind = np.delete(indx.n, indx.elbow)[
                    delta1 > eps2]  # Remove Elbow from observations and keep non-zero elements
                tmp_delta = delta1[delta1 > eps2]  # All deltas that are non-zero
                i_star = tmp_ind[tmp_delta.argmin()]
                logger.debug(f'i_star = {i_star}')

            # Update beta
            coef.beta0[idx] = coef.beta0[idx - 1] + delta * coef.nu0
            coef.beta[idx] = coef.beta[idx - 1]
            coef.beta[idx, indx.active] = coef.beta[idx - 1, indx.active] + delta * coef.nu

            logger.debug(f'Update intercept from {coef.beta0[idx - 1, 0]:.2f} to {coef.beta0[idx, 0]:.2f}')
            logger.debug(f'Update slope from\n{coef.beta[idx - 1, indx.active]} to\n{coef.beta[idx, indx.active]}')

            if coef.s[idx] > s_max:
                logger.info(f's = {coef.s[idx]:.2f} >= {s_max:.2f} is large enough')
                break

            # Reduce residuals not in the elbow by delta*gam
            coef.residual[np.in1d(indx.n, indx.elbow, invert=True)] -= delta * gam

            # Check if there are points in either L or R if we do not drop
            if (indx.left.size + indx.right.size == 1) & (not drop):
                logger.info('No point in Left or Right')
                break

            # Add a variable to the active set
            # Test if all variables are included. If yes, set lambda_var to -inf and continue with next step
            if indx.active.size == k:
                coef.lambda_var = np.full((2, k), -np.inf)
            else:
                indx.inactive = indx.k[np.in1d(indx.k, indx.active, invert=True)]  # All variables not in V
                indx.tmp_elbow = indx.elbow
                indx.tmp_left = indx.left
                indx.tmp_right = indx.right

                if drop:
                    indx.active = indx.active[indx.active != j1]  # Remove the detected variable from V
                else:
                    # Add i_star to the Elbow and remove it from either Left or Right
                    # (we know that i_star hits the elbow)
                    logger.debug(f'Move {i_star} from Left/Right to Elbow')
                    indx.tmp_elbow = np.append(indx.tmp_elbow, i_star)
                    indx.tmp_left = indx.tmp_left[indx.tmp_left != i_star]
                    indx.tmp_right = indx.tmp_right[indx.tmp_right != i_star]

                coef.lambda_var = np.zeros((2, indx.inactive.size))  # First row: sign=1, second row: sign=-1
                coef.lambda_var[coef.lambda_var == 0] = -np.inf  # Initially set to -inf (want to maximize lambda)
                coef.nu_var = np.zeros((2, indx.inactive.size, 1 + indx.active.size + 1))  # Store nus in 3d array
                coef.b = np.array([0] * (indx.active.size + 1) + [1])  # The 1_0 vector (see p. 171 bottom)

                for j_idx in range(indx.inactive.size):
                    j_star = indx.inactive[j_idx]  # Select variable j as candidate for the next active variable

                    # Select all columns of x that are in indx.active and additionally j_star.
                    # Transposition improves performance as Python stores array in row-major order
                    x_v = coef.xc.T.take(np.append(0, np.append(indx.active, j_star) + 1), axis=0, mode='clip').T

                    # Combination of (2.10) and (2.11)
                    x0 = np.vstack((np.hstack((np.ones((indx.tmp_elbow.size, 1)),
                                               self.x[indx.tmp_elbow][:, indx.active].reshape((indx.tmp_elbow.size, -1)),
                                               self.x[indx.tmp_elbow, j_star].reshape((indx.tmp_elbow.size, -1)))),
                                    np.hstack(
                                        (0, np.sign(coef.beta[idx, indx.active]), np.nan))))  # nan is a placeholder for sign

                    # Sign of the next variable to include may be either positive or negative
                    for sign in (1, -1):
                        index = np.where(sign == 1, 0, 1)  # Index in nu_var and lambda_var
                        x0[-1, -1] = sign  # Change sign in the x0 matrix

                        try:
                            nu_tmp = np.linalg.solve(x0, coef.b)  # Solve system (p. 171 bottom)

                            # If sign of last entry != sign then leave at -inf.
                            if np.sign(nu_tmp[-1]) == sign:
                                coef.nu_var[index, j_idx, :] = nu_tmp
                                # Store sets that are used to compute -lambda* (p. 172))
                                x_l = x_v.take(indx.tmp_left, axis=0, mode='clip')
                                x_r = x_v.take(indx.tmp_right, axis=0, mode='clip')
                                coef.lambda_var[index, j_idx] = -((1 - self.alpha) * np.dot(x_l, nu_tmp).sum() -
                                                                  self.alpha * np.dot(x_r, nu_tmp).sum())
                        except LinAlgError:
                            pass

                # Select the maximum of each column
                coef.nu_var = coef.nu_var[coef.lambda_var.argmax(axis=0), np.arange(indx.inactive.size), :]
                coef.lambda_var = coef.lambda_var.max(axis=0)

            # Remove an observation from the elbow
            lambda_obs = np.zeros(indx.tmp_elbow.size)
            lambda_obs[lambda_obs == 0] = -np.inf
            nu_obs = np.zeros((1 + indx.active.size, indx.tmp_elbow.size))
            left_obs = np.zeros(indx.tmp_elbow.size)  # 1 if if we shifted observation to the left
            coef.b = np.array([0] * indx.active.size + [1])

            # Store the L and the R observations of x
            x_v = coef.xc.T.take(np.append(0, indx.active + 1), axis=0, mode='clip').T
            x_r = x_v.take(indx.tmp_right, axis=0, mode='clip')
            x_l = x_v.take(indx.tmp_left, axis=0, mode='clip')

            # Combination of (2.10) and (2.11), here without an additional variable j
            x0_all = np.vstack((np.hstack((np.ones((indx.tmp_elbow.size, 1)),
                                           self.x[indx.tmp_elbow][:, indx.active].reshape((indx.tmp_elbow.size, -1)))),
                                np.hstack((0, np.sign(coef.beta[idx, indx.active])))))

            for i in range(indx.tmp_elbow.size):
                x0 = np.delete(x0_all, i, 0)  # Delete the ith observation
                try:
                    nu_tmp = np.linalg.solve(x0, coef.b)  # Solve system (p. 171 bottom)
                    nu_obs[:, i] = nu_tmp
                    # Save lambda achieved by removing observation i
                    lambda_obs[i] = -((1 - self.alpha) * np.dot(x_l, nu_tmp).sum() -
                                      self.alpha * np.dot(x_r, nu_tmp).sum())

                    # Test if we shift i to the left or to the right
                    tmpyf = np.dot(np.append(1, self.x[indx.tmp_elbow[i], indx.active]), nu_obs[:, i])
                    if tmpyf > 0:  # To the left
                        left_obs[i] = 1
                        lambda_obs[i] += -(1 - self.alpha) * tmpyf
                    else:  # To the right
                        lambda_obs[i] += self.alpha * tmpyf
                except LinAlgError:
                    pass

            # Compare the effects of adding one variable to V and removing one observation from E
            lam_var = coef.lambda_var.max()  # Maximum lambda from adding a variable
            lam_obs = lambda_obs.max()  # Maximum lambda from removing an observation from E

            if (lam_var > lam_obs) & (lam_var > 0):  # Add variable to V
                lam = lam_var

                indx.active = np.append(indx.active, indx.inactive[coef.lambda_var.argmax()])

                if not drop:  # Add i_star to the elbow
                    logger.debug(f'Move {i_star} from Left/Right to Elbow')
                    indx.elbow = np.append(indx.elbow, i_star)
                    indx.left = indx.left[indx.left != i_star]
                    indx.right = indx.right[indx.right != i_star]

                # Store nu0 and nu
                coef.nu0 = coef.nu_var[coef.lambda_var.argmax(), 0]
                coef.nu = coef.nu_var[coef.lambda_var.argmax(), 1:]

            elif (lam_obs > lam_var) & (lam_obs > 0):  # Remove observation from E
                lam = lam_obs

                # i_star remains in E, no change in V
                if not drop:
                    logger.debug(f'Remove {i_star} from Left/Right')
                    indx.left = indx.left[indx.left != i_star]
                    indx.right = indx.right[indx.right != i_star]

                # Find a new i_star
                i_star = indx.tmp_elbow[lambda_obs.argmax()]
                logger.debug(f'New i_star = {i_star}')
                coef.nu0 = nu_obs[0, lambda_obs.argmax()]
                coef.nu = nu_obs[1:, lambda_obs.argmax()]

                # Test if we need to add i_star to L or R
                if left_obs[lambda_obs.argmax()] == 1:
                    logger.debug(f'Append {i_star} to Left')
                    indx.left = np.append(indx.left, i_star)
                else:
                    logger.debug(f'Append {i_star} to Right')
                    indx.right = np.append(indx.right, i_star)

                # Remove i_star from E
                logger.debug(f'Removing {i_star} from the Elbow')
                indx.elbow = indx.tmp_elbow[indx.tmp_elbow != i_star]
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
        self.beta0 = coef.beta0[:idx][:, 0]
        self.beta = coef.beta[:idx]
        self.s = coef.s[:idx]
        self.var_names = self.var_names

        # Save the interpolated estimates
        s = np.linspace(self.s[0], self.s[-1], 1000)               # Interpolate shrinkage values
        b0 = pd.Series(np.interp(s, self.s, self.beta0), index=s)  # Extract and interpolate intercept
        b = pd.DataFrame(np.apply_along_axis(lambda w: np.interp(s, self.s, w), 0, self.beta),
                         index=s, columns=self.var_names)          # Extract and interpolate slope
        self.b0 = b0
        self.b = b

    def _assign_first_variable(self, coef, indx):
        for j_idx, j_star in enumerate(indx.inactive):
            x_v = coef.xc[:, np.append(0, j_star + 1)]

            # Sign of the next variable to include may be either positive or negative
            for sign in (1, -1):
                index = np.where(sign == 1, 0, 1)  # Index in nu_var and lambda_var

                # Combination of (2.10) and (2.11)
                x0 = np.vstack((np.hstack((1, np.mat(self.x)[indx.tmp_elbow, j_star])), np.hstack((0, sign))))

                try:  # Check if x0 has full rank
                    nu_tmp = np.linalg.solve(x0, coef.b)  # Solve system (p. 171 bottom)
                    coef.nu_var[index, j_idx, :] = nu_tmp

                    # Store sets that are used to compute -lambda* (p. 172)
                    x_l = x_v.take(indx.tmp_left, axis=0, mode='clip')
                    x_r = x_v.take(indx.tmp_right, axis=0, mode='clip')

                    # Save lambda achieved by the current variable. If sign of last entry != sign then leave at -inf.
                    if np.sign(nu_tmp[-1]) == sign:
                        coef.lambda_var[index, j_idx] = -((1 - self.alpha) * np.dot(x_l, nu_tmp).sum() -
                                                          self.alpha * np.dot(x_r, nu_tmp).sum())
                except LinAlgError:
                    logger.debug(f'sign: {sign}')
        # Select the nu corresponding to the maximum lambda and store the maximum lambda
        coef.nu_var = coef.nu_var[coef.lambda_var.argmax(axis=0), np.arange(indx.inactive.size), :]
        coef.lambda_var = coef.lambda_var.max(axis=0)
        # Store the active variable
        indx.active = indx.inactive[coef.lambda_var.argmax()]
        # Store initial nu0 and nu
        coef.nu0 = coef.nu_var[indx.active, 0]
        coef.nu = coef.nu_var[indx.active, 1:]

        return indx, coef

    def _get_initial_beta0(self, n):
        """
        There are actually two cases, first if n*tau is integer, second if tau*n is non-integer.
        Here I assume that in the latter case all the weight is on the first component (see section 2.2)
        """
        _initial_beta0 = np.sort(self.y)[int(np.floor(self.alpha * n))]
        logger.info(f'Finding initial solution: beta0 = {_initial_beta0:.2f}')
        return _initial_beta0

    def _check_if_y_can_be_ordered(self, n):
        y_can_be_ordered_strictly = np.unique(self.y).size != n
        if y_can_be_ordered_strictly:
            logger.info('Adding noise to y because y contains duplicate values')
            self.y += np.random.normal(loc=0, scale=10 ** -5, size=self.y.size)

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


if __name__ == "__main__":

    # Test data: the daily log returns of the IBM stock and the 1% VaR forecasts stemming from a variety of risk models.
    Y = pd.read_csv("input/returns.txt", index_col=0).squeeze()
    X = pd.read_csv("input/quantile_predicitons.txt", index_col=0)

    # Estimate the lasso quantile regression
    mdl = L1QR(y=Y, x=X, alpha=0.01)
    mdl.fit(s_max=3)
    mdl.plot_trace(file_name="output/trace_plot.png", size=(10, 6))

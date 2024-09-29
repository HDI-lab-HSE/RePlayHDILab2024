from obp.ope import InverseProbabilityWeighting
from dataclasses import dataclass
import numpy as np
import itertools
from typing import Dict
from typing import Optional 
from obp.ope.helper import estimate_bias_in_ope
from obp.ope.helper import estimate_high_probability_upper_bound_bias


@dataclass
class Exp_Smooth_IPS_Min(InverseProbabilityWeighting):
    """Self-Normalized Inverse Probability Weighting (SNIPW) Estimator.

    Note
    -------
    SNIPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{n} [w(x_i,a_i) r_i]}{ \\mathbb{E}_{n} [w(x_i,a_i)]},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    SNIPW normalizes the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and gains some stability in OPE.
    See the reference papers for more details.

    Parameters
    ----------
    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='snipw'.
        Name of the estimator.

    References
    ----------
    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "ESIPSMIN"
    beta: float = 0.9

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * (iw ** self.beta)
    
    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        use_bias_upper_bound: bool, default=True
            Whether to use a bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound used in SLOPE.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of IPW with clipping
        iw_hat = self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        sample_variance = np.var(
            iw_hat
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of IPW with clipping
        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward, iw=iw, iw_hat=iw_hat, delta=delta
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score
    
    
@dataclass
class Exp_Smooth_IPS_Max(InverseProbabilityWeighting):
    """Self-Normalized Inverse Probability Weighting (SNIPW) Estimator.

    Note
    -------
    SNIPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{n} [w(x_i,a_i) r_i]}{ \\mathbb{E}_{n} [w(x_i,a_i)]},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    SNIPW normalizes the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and gains some stability in OPE.
    See the reference papers for more details.

    Parameters
    ----------
    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='snipw'.
        Name of the estimator.

    References
    ----------
    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "ESIPSMAX"
    beta: float = 0.9

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist[np.arange(action.shape[0]), action, position] / (pscore ** self.beta)
        return reward * iw 
    
    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        use_bias_upper_bound: bool, default=True
            Whether to use a bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound used in SLOPE.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of IPW with clipping
        iw_hat = self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        sample_variance = np.var(
            iw_hat
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of IPW with clipping
        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward, iw=iw, iw_hat=iw_hat, delta=delta
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score
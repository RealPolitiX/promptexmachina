from .texter import TextAggregator
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Sequence, Union
import warnings

import torch
from nltk.tokenize import sent_tokenize

# general BoTorch imports
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# BoTorch uniobjective
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement

# BoTorch multiobjective
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

# optuna imports
from optuna import logging
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler, IntersectionSearchSpace, RandomSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

# persona
persona_dict = {}


def set_logging(level):

    if level == 'warning':
        logging.set_verbosity(logging.WARNING)
    elif logging == 'all':
        logging.set_verbosity(logging.INFO)
    elif logging == 'nothing':
        from warnings import simplefilter
        simplefilter("ignore", category=RuntimeWarning)


def qei_candidates_func(train_x: "torch.Tensor", train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"], bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"], acqf_dict:Dict={}) -> "torch.Tensor":
    """Quasi MC-based batch Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        if train_obj_feas.numel() == 0:
            # TODO(hvy): Do not use 0 as the best observation.
            best_f = torch.zeros(())
        else:
            best_f = train_obj_feas.max()

        constraints = []
        n_constraints = train_con.size(1)
        for i in range(n_constraints):
            constraints.append(lambda Z, i=i: Z[..., -n_constraints + i])
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=constraints,
        )
    else:
        train_y = train_obj

        best_f = train_obj.max()

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf_type = acqf_dict.pop('acqf_type', 'qmc')
    if acqf_type == 'qmc':
        acqf = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512])),
            objective=objective,
            X_pending=pending_x,
        )
    elif acqf_type == 'analytic':
        acqf = ExpectedImprovement(
            model=model,
            best_f = best_f,
        )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # Unpack parameters for the acquisition function
    acqf_bounds = acqf_dict.pop('bounds', standard_bounds)
    q = acqf_dict.pop('q', 1)
    num_restarts = acqf_dict.pop('num_restarts', 1)
    raw_samples = acqf_dict.pop('raw_samples', 10)
    maxiter = acqf_dict.pop('maxiter', 100)
    sequential = acqf_dict.pop('sequential', True)
    options = acqf_dict or {} # Leave the rest for options
    
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=acqf_bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        sequential=sequential,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def qnei_candidates_func(train_x: "torch.Tensor", train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"], bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"], acqf_dict:Dict={}) -> "torch.Tensor":
    """Quasi MC-based batch Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        # if train_obj_feas.numel() == 0:
        #     # TODO(hvy): Do not use 0 as the best observation.
        #     best_f = torch.zeros(())
        # else:
        #     best_f = train_obj_feas.max()

        constraints = []
        n_constraints = train_con.size(1)
        for i in range(n_constraints):
            constraints.append(lambda Z, i=i: Z[..., -n_constraints + i])
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=constraints,
        )
    else:
        train_y = train_obj

        # best_f = train_obj.max()


        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512])),
        objective=objective,
        X_pending=pending_x,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    # Unpack parameters for the acquisition function
    acqf_bounds = acqf_dict.pop('bounds', standard_bounds)
    q = acqf_dict.pop('q', 1)
    num_restarts = acqf_dict.pop('num_restarts', 1)
    raw_samples = acqf_dict.pop('raw_samples', 10)
    maxiter = acqf_dict.pop('maxiter', 100)
    sequential = acqf_dict.pop('sequential', True)
    options = acqf_dict or {} # Leave the rest for options
    
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=acqf_bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        sequential=sequential,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def qehvi_candidates_func(train_x: "torch.Tensor", train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"], bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"], acqf_dict:Dict={},) -> "torch.Tensor":
    """Quasi MC-based batch Expected Hypervolume Improvement (qEHVI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is three or less.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        constraints = []
        n_constraints = train_con.size(1)

        for i in range(n_constraints):
            constraints.append(lambda Z, i=i: Z[..., -n_constraints + i])
        
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": constraints,
        }
    else:
        train_y = train_obj
        train_obj_feas = train_obj
        additional_qehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj_feas, alpha=alpha)

    ref_point_list = ref_point.tolist()

    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512])),
        X_pending=pending_x,
        **additional_qehvi_kwargs,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    
    # Unpack parameters for the acquisition function
    acqf_bounds = acqf_dict.pop('bounds', standard_bounds)
    q = acqf_dict.pop('q', 1)
    num_restarts = acqf_dict.pop('num_restarts', 1)
    raw_samples = acqf_dict.pop('raw_samples', 10)
    maxiter = acqf_dict.pop('maxiter', 100)
    sequential = acqf_dict.pop('sequential', True)
    options = acqf_dict or {} # Leave the rest for options
    
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=acqf_bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        sequential=sequential,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def qnehvi_candidates_func(train_x: "torch.Tensor", train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"], bounds: "torch.Tensor",
    pending_x: Optional["torch.Tensor"], acqf_dict:Dict={},) -> "torch.Tensor":
    """Quasi MC-based batch Expected Hypervolume Improvement (qEHVI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is three or less.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        n_constraints = train_con.size(1)
        additional_qnehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": [
                (lambda Z, i=i: Z[..., -n_constraints + i]) for i in range(n_constraints)
            ],
        }
    else:
        train_y = train_obj

        additional_qnehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)
    if pending_x is not None:
        pending_x = normalize(pending_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
    if n_objectives > 4:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    ref_point_list = ref_point.tolist()

    # prune_baseline=True is generally recommended by the documentation of BoTorch.
    # cf. https://botorch.org/api/acquisition.html (accessed on 2022/11/18)
    acqf = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        alpha=alpha,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256])),
        X_pending=pending_x,
        **additional_qnehvi_kwargs,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    
    # Unpack parameters for the acquisition function
    acqf_bounds = acqf_dict.pop('bounds', standard_bounds)
    q = acqf_dict.pop('q', 1)
    num_restarts = acqf_dict.pop('num_restarts', 1)
    raw_samples = acqf_dict.pop('raw_samples', 10)
    maxiter = acqf_dict.pop('maxiter', 100)
    sequential = acqf_dict.pop('sequential', True)
    options = acqf_dict or {} # Leave the rest for options
    
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=acqf_bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        sequential=sequential,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


class PromptObjective:

    def __init__(self, max_sents, max_elems):

        self.max_sents = max_sents
        self.max_elems = max_elems


class SinglePersonaObjective(PromptObjective):

    def __init__(self, topic, p_constructor, p_evaluator, eval_kwargs,
                 max_elems=1, max_sents=None):
        super().__init__(max_sents, max_elems)

        topic_ref = {1:'biology', 2:'chemistry', 3:'medicine',
          4:'physics', 5:'math', 6:'computer_science'}
        topic_ref = dict([(v, k) for k, v in topic_ref.items()])
        self.topic_id = topic_ref[topic]
        self.p_constructor = p_constructor
        self.p_evaluator = p_evaluator
        self.eval_kwargs = eval_kwargs

    def __call__(self, trial):
        
        pers_id = trial.suggest_int('pers_id', 1, 6, step=1)
        pers_len = trial.suggest_int('pers_len', 0, 8, step=1)

        return self.p_evaluator(pers_id, pers_len, self.topic_id-1)


class MultiPersonaObjective(PromptObjective):

    def __init__(self, topic, p_constructor, p_evaluator, eval_kwargs,
                 max_elems, max_sents=None):
        super().__init__(max_sents, max_elems)
        
        topic_ref = {1:'biology', 2:'chemistry', 3:'medicine',
          4:'physics', 5:'math', 6:'computer_science'}
        topic_ref = dict([(v, k) for k, v in topic_ref.items()])
        self.topic_id = topic_ref[topic]
        self.p_constructor = p_constructor
        self.p_evaluator = p_evaluator
        self.eval_kwargs = eval_kwargs

    def __call__(self, trial):

        pers_ids = [trial.suggest_int('pers_id_{}'.format(i+1), 1, 6, step=1) for i in range(self.max_elems)]
        pers_len = trial.suggest_int('pers_len', 0, 8, step=1)

        return self.p_evaluator(pers_ids, pers_len, self.topic_id-1)



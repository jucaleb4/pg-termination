""" Basic stochastic PMD """
import time
import os
import warnings
from enum import IntEnum
import multiprocessing as mp

import numpy as np

from pg_termination import pmd
from pg_termination import mdpmodel 
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10
MSG_1 = "We changed 'pi_threshold'->'pi_threshold_mult'.Please update the yaml file (defaulting to 'pi_threshold_mult=1')"

def policy_update(pi, psi, eta, theta, is_finite_state):
    """ Closed-form solution with KL 

    # TODO: Add other divergences, e.g., Tsallis...
    """
    (n_states, n_actions) = psi.shape

    assert (not np.any(np.isnan(psi))), "Found NaN in psi, quitting program"
    pi *= np.exp(-eta*(psi - np.outer(np.min(psi, axis=1), np.ones(n_actions)))).T
    pi /= np.outer(np.ones(n_actions), np.sum(pi, axis=0))

def policy_validation(env, pi, settings):
    """
    Evaluates upper policy value V(pi) and V(pi-star).

    :return agg_V: value function (upper bound)
    :return agg_V_star: optimal value (lower bound)
    :return agg_err: uniform error on avg_V to true_V
    :return avg_total_err: averaged (over t) uniform error on V_t to true_V
    """
    agg_psi = np.zeros((env.n_states, env.n_actions), dtype=float)
    agg_V = np.zeros(env.n_states, dtype=float)
    total_V_err = 0.

    true_V = -np.inf
    if not settings["skip_true_model"]:
        (_, true_V) = env.get_advantage(pi)

    # TODO: Use `early_terminate` value returned by @estimating_mixing_properties and @estimate_advantage_online_mc_dynamic
    for i in range(settings["validation_k"]):
        if settings["estimate_Q"] == "generative":
            (psi, V, _) = env.estimate_advantage_generative(pi, settings["N_mc"], settings["T_mc"])
        elif settings["estimate_Q"] == "online": # @depreciated
            (_, psi, V, _, _) = env.estimate_advantage_online_mc(pi, settings["T_mc"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_fixed":
            (_, psi, V, _, _) = env.estimate_advantage_online_mc(pi, settings["T_mc"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_estimate":
            tmix, nu = env.get_mixing_time_ub(pi)
            (_, nu_est, tmix_est, _) = env.estimate_mixing_properties(pi, 0, tmix=tmix, nu=nu)
            # based from Proposition 5.3 from https://arxiv.org/abs/2303.04386
            T = int(1./(1-env.gamma) + (tmix_est*env.n_actions)/(np.min(nu_est)*(1.-env.gamma)) + 1)
            (_, psi, V, _, _) = env.estimate_advantage_online_mc(pi, T, settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_dynamic":
            (_, psi, V, _, _) = env.estimate_advantage_online_mc_dynamic(pi, settings["pi_threshold"])
        elif settings["estimate_Q"] == "ctd":
            # during validation, use online model
            (_, psi, V, _, _) = env.estimate_advantage_online_mc_dynamic(pi, settings["pi_threshold"])

        agg_psi += psi
        agg_V += V
        total_V_err += np.max(np.abs(V - true_V))

    N = max(1, float(settings["validation_k"])) # avoid zero

    agg_psi /= N
    agg_V /= N
    agg_V_err = np.max(np.abs(agg_V - true_V))
    avg_total_V_err = total_V_err / N

    print("Agg V err:   %.2e | Avg point V err: %.2e" % (agg_V_err, avg_total_V_err))

    return agg_psi, agg_V, agg_V_err, avg_total_V_err

def get_loggers(settings):
    seed = settings['seed']
    env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed)
    logger = BasicLogger(
        fname=os.path.join(settings["log_folder"], "seed=%d.csv" % seed), 
        keys=["iter", "point value", "point opt_lb", "point uni_opt_lb", 
              "agg value", "agg opt_lb", "agg uni_opt_lb", "true value", 
              "true opt_lb", "true uni_opt_lb"], 
        dtypes=['d'] + ['f'] * 9
    )
    logger_agg_V = BasicLogger(
        fname=os.path.join(settings["log_folder"], "agg_V_seed=%d.csv" % seed), 
        keys=["iter"] + ["s_%d" % s for s in range(env.n_states)],
        dtypes=['d'] + ['f'] * env.n_states,
    )
    logger_agg_advgap = BasicLogger(
        fname=os.path.join(settings["log_folder"], "agg_advgap_seed=%d.csv" % seed), 
        keys=["iter"] + ["s_%d" % s for s in range(env.n_states)],
        dtypes=['d'] + ['f'] * env.n_states,
    )
    logger_validation = BasicLogger(
        fname=os.path.join(settings["log_folder"], "validation_seed=%d.csv" % seed), 
        keys=["agg value", "agg opt_lb", "agg uni_opt_lb", "true value", "true opt_lb", "true uni_opt_lb", "agg V_err", "avg total_V_err"],

        dtypes=['f'] * 8,
    )
    logger_mixing = BasicLogger(
        fname=os.path.join(settings["log_folder"], "mixing_seed=%d.csv" % seed),  
        keys=["iter", "cum_time", "cum_samples", "cum_est_samples"] + ["nu_lb", "t_mix"],
        dtypes=['d', 'f', 'd', 'd'] + ['f'] * 2,
    )

    return [logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing]

def _train_with_tuning(settings):
    print("=== Running tuning via successive halving ===")
    logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing = get_loggers(settings)
    seed = settings['seed']

    logger_tune = BasicLogger(
        fname=os.path.join(settings["log_folder"], "tuning_map_seed=%d.csv" % seed),  
        keys=["round","tune_id", "n_iter", "explore_T"],
        dtypes=['d', 'd', 'd', 'd'],
    )

    rng = np.random.default_rng(seed)
    num_trials = settings['successive_half_trials']
    n_iters = settings["n_iters"]
    # define n_iters so that total successive halving does is approximately n_iters
    sub_n_iters = max(1, int(n_iters/(np.log(num_trials)/np.log(2))))
    settings["n_iters"] = sub_n_iters

    log_min_T = np.log(settings['min_T_mc'])/np.log(10)
    log_max_T = np.log(settings['max_T_mc'])/np.log(10)
    Ts = np.power(10, rng.uniform(low=log_min_T, high=log_max_T, size=num_trials)).astype('int')
    tune_id_arr = np.arange(num_trials)
    tune_round = 0
    Pi_arr = None
    while 1:
        score_arr = np.zeros(len(Ts))
        # run all trials
        for i, T in enumerate(Ts):
            settings['T_mc'] = T
            pi_0 = None if tune_round == 0 else Pi_arr[i]
            pi_t, f_t = _spmd(settings, 1.0, logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing, pi_0=pi_0)
            if Pi_arr is None:
                Pi_arr = np.zeros((len(Ts),) + pi_t.shape)
            Pi_arr[i,:,:] = pi_t
            score_arr[i] = f_t

            logger_tune.log(tune_round, tune_id_arr[i], sub_n_iters, T)

        # save progress and update
        if len(Pi_arr) <= 1:
            break
        # keep half the best
        sorted_idx = np.argsort(score_arr)
        sorted_idx = sorted_idx[:int(len(sorted_idx)/2)]
        Pi_arr = Pi_arr[sorted_idx]
        Ts = Ts[sorted_idx]
        tune_id_arr = tune_id_arr[sorted_idx]
        tune_round += 1

    logger.save()
    logger_mixing.save()
    logger_agg_V.save()
    logger_agg_advgap.save()
    logger_validation.save()
    logger_tune.save()

def _train(settings):
    logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing = get_loggers(settings)

    # TODO: Make this automated
    ukappa = (1.-settings['gamma'])**(-2)
    _spmd(settings, ukappa, logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing)

    logger.save()
    logger_mixing.save()
    logger_agg_V.save()
    logger_agg_advgap.save()
    logger_validation.save()

def policy_eval(env, settings, pi, tmix, unu, Phi, ukappa, is_finite_state, time_limit=np.inf):
    """ Policy evaluation
    :param env: environment from mdpmodel 
    :param settings: (dict) 
    :param pi: policy
    :param tmix: (only for TOMC) mixing time
    :param unu: (only for TOMC) minimum stationary dist
    :param Phi: (only for CTD) features
    :param ukappa: (only for CTD) estimate of minimum of kappa (distribution)
    :param is_finite_state: (only for CTD, boolean)
    """
    theta = None
    n_est_samples = 0
    s_time = time.time()
    early_terminate = False

    if settings["estimate_Q"] == "generative":
        (psi, V, n_samples) = env.estimate_advantage_generative(pi, settings["N_mc"], settings["T_mc"])
    elif settings["estimate_Q"] == "online": # @depreciated
        (early_terminate, psi, V, _, n_samples) = env.estimate_advantage_online_mc(pi, settings["T_mc"], settings["pi_threshold"], time_limit)
    elif settings["estimate_Q"] == "online_mc_fixed":
        (early_terminate, psi, V, _, n_samples) = env.estimate_advantage_online_mc(pi, settings["T_mc"], settings["pi_threshold"], time_limit)
    elif settings["estimate_Q"] == "online_mc_estimate":
        (early_terminate, nu_est, tmix_est, n_est_samples) = env.estimate_mixing_properties(pi, 0, tmix=tmix, nu=unu, time_limit=time_limit/2)
        if early_terminate:
            return (early_terminate, None, None, None, None, None)
        T = int(1./(1-env.gamma) + (tmix_est*env.n_actions)/(np.min(nu_est)*(1.-env.gamma)) + 1)
        time_left_adj = max(time_limit/2, time_limit - (time.time()-s_time))
        (early_terminate, psi, V, _, n_samples) = env.estimate_advantage_online_mc(pi, T, settings["pi_threshold"], time_limit=time_left_adj)
    elif settings["estimate_Q"] == "online_mc_dynamic":
        (early_terminate, psi, V, _, n_samples) = env.estimate_advantage_online_mc_dynamic(pi, settings["eps"], settings["pi_threshold"], time_limit=time_limit)
    elif settings["estimate_Q"] == "ctd": 
        # pass in theta as last argument to warm start (doesn't help too much)
        (psi, V, n_samples, theta) = env.estimate_advantage_online_ctd(
            pi, Phi, ukappa, settings["eps"], settings["delta"], 
            settings['ctd_iota_mult'], is_finite_state, 
        )
    else: 
        raise Exception("Unknown estimate_Q setting %s" % settings["estimate_Q"])

    return (early_terminate, psi, V, n_samples, n_est_samples, theta) 

def print_spmd_progress(env, t, V_t, psi_t, agg_V_t, agg_psi_t, true_V_t, true_psi_t, logger):
    """ Print and updates SPMD certificates.

    Inputs @agg_V_t and @agg_psi_t and over-written in place.

    :param env: environment from mdpmodel 
    :param t: iteration of SPMD
    :param V_t: estimated value func at iter t
    :param psi_t: estimated advantage func at iter t
    :param agg_V_t: aggregated estimate of value func at iter t
    :param agg_psi_t: aggregated estimate of advantage func at iter t
    :param true_V_t: (only TOMC, o/w arbitrary) true value func at iter t
    :param true_psi_t: (only TOMC, o/w arbitrary) true advantage func at iter t
    :param is_finite_state: (only CTD, boolean)
    :param logger: 
    """
    alpha_t = 1./(t+1)
    agg_psi_t[:,:] = (1.-alpha_t)*agg_psi_t + alpha_t*psi_t
    agg_V_t[:] = (1.-alpha_t)*agg_V_t + alpha_t*V_t
    if ((t+1) <= 100 and (t+1) % 5 == 0) or (t+1) % 100==0 or t<=10:
        print("Iter %d: f=%.2e (fstar_lb=%.2e) | ag_f=%.2e (ag_fstar_lb=%.2e) | true_f=%.2e (true_fstar_lb=%.2e)" % (
            t+1, 
            np.dot(env.rho, V_t), 
            np.dot(env.rho, V_t - np.max(-psi_t, axis=1)/(1.-env.gamma)), 
            np.dot(env.rho, agg_V_t), 
            np.dot(env.rho, agg_V_t - np.max(-agg_psi_t, axis=1)/(1.-env.gamma)), 
            np.dot(env.rho, true_V_t), 
            np.dot(env.rho, true_V_t - np.max(-true_psi_t, axis=1)/(1.-env.gamma)),
        ))

    logger.log(
        t+1, 
        np.dot(env.rho, V_t), 
        np.dot(env.rho, V_t - np.maximum(0, np.max(-psi_t, axis=1)/(1.-env.gamma))), 
        np.dot(env.rho, V_t - np.max(-psi_t)/(1.-env.gamma)), 
        np.dot(env.rho, agg_V_t), 
        np.dot(env.rho, agg_V_t - np.maximum(0, np.max(-agg_psi_t, axis=1)/(1.-env.gamma))), 
        np.dot(env.rho, agg_V_t - np.max(-agg_psi_t)/(1.-env.gamma)), 
        np.dot(env.rho, true_V_t), 
        np.dot(env.rho, true_V_t - np.maximum(0, np.max(-true_psi_t, axis=1)/(1.-env.gamma))),
        np.dot(env.rho, true_V_t - np.max(-true_psi_t)/(1.-env.gamma)),
    )

    return agg_V_t, agg_psi_t

def _spmd(settings, ukappa, logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing, pi_0=None):
    """
    SPMD training procedure 
    
    :param settings: dictionary of all user-defined parameter values
    :param ukappa: estimation of ukappa (if using CTD and not set, will set warning and default to 0)
    """
    assert ukappa > 0, "ukappa=%.4e not positive" % ukappa
    seed = settings['seed']

    # initialization
    env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed)
    if pi_0 is None:
        pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)
    next_greedy_pi_t = np.copy(pi_0)
    pi_star = None
    agg_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    agg_V_t = np.zeros(env.n_states, dtype=float)
    true_psi = true_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    true_V = true_V_t = np.zeros(env.n_states, dtype=float)
    cum_samples = cum_est_samples = 0
    tmix = unu = 0
    Phi = None
    is_finite_state = env.n_states is not None
    s_time = time.time()
    e_time = -time.time()
    last_V_t = np.inf * np.ones(len(env.rho))

    stepsize_scheduler = pmd.StepsizeSchedule(env, settings["stepsize_rule"], 
                                              settings.get("eta",1))
    if "pi_threshold_mult" not in settings:
        warnings.warn(MSG_1)
        settings["pi_threshold_mult"] = 1.
    settings["pi_threshold"] = settings["pi_threshold_mult"] * (1.-env.gamma)/env.n_actions

    if settings["estimate_Q"] == "ctd": 
        if is_finite_state: # finite state and action
            Phi = env.rng.normal(size=(env.n_states*env.n_actions, settings["ctd_feature_size"]))
        else: # finite action and continuous state
            Phi = env.rng.normal(size=(env.n_states, env.n_state_dim, settings["ctd_feature_size"]))
        theta = np.zeros(settings["ctd_feature_size"])

    # SPMD main for-loop
    for t in range(settings["n_iters"]):
        if not settings["skip_true_model"]:
            tmix, nu = env.get_mixing_time_ub(pi_t)
            unu = np.min(nu)
            e_time += time.time() # skip advantage estimation time
            (true_psi_t, true_V_t) = env.get_advantage(pi_t)
            e_time -= time.time()

        time_left = (settings["max_runtime_in_sec"] - (time.time() - s_time)) * 1.05
        (early_terminate, psi_t, V_t, n_samples, n_est_samples, theta_t) = policy_eval(
                env, settings, pi_t, tmix, unu, Phi, ukappa, is_finite_state, time_left
        )
        if early_terminate: 
            print("=== Breaking early because we predicted exceeding the max runtime ===")
            break
        eta_t = stepsize_scheduler.get_stepsize(t, psi_t)
        policy_update(pi_t, psi_t, eta_t, theta_t, is_finite_state) 
        
        cum_samples += n_samples
        cum_est_samples += n_est_samples
        print_spmd_progress(env, t, V_t, psi_t, agg_V_t, agg_psi_t, true_V_t, true_psi_t, logger)
        logger_mixing.log(t, e_time + time.time(), cum_samples, cum_est_samples, unu, tmix)
        last_V_t = V_t

        total_runtime = time.time() - s_time
        if total_runtime >= settings["max_runtime_in_sec"]:
            print("=== Breaking early because we exceeded the max runtime ===")
            break

    # policy validation
    output = policy_validation(env, pi_t, settings)
    (agg_psi, agg_V, agg_V_err, avg_total_V_err) = output
    alpha = float(settings["validation_k"])/(settings["validation_k"] + settings["n_iters"])
    double_agg_V = alpha*agg_V + (1.-alpha)*agg_V_t
    double_agg_psi = alpha*agg_psi + (1.-alpha)*agg_psi_t

    # logger_agg_adv.log(t+1, *list(double_agg_psi.ravel()))
    logger_agg_V.log(t+1, *list(agg_V_t))
    double_agg_advgap = np.maximum(0, np.max(-agg_psi_t, axis=1))
    logger_agg_advgap.log(t+1, *list(double_agg_advgap))

    if not settings["skip_true_model"]:
        (true_psi, true_V) = env.get_advantage(pi_t)

    print("Total runtime: %.2fs" % (time.time() - s_time))
    print("Offline: f=%.2e (fstar=%.2e) | true_f=%.2e (est_true_f_star=%.2e)" % (
        np.dot(env.rho, agg_V), 
        np.dot(env.rho, double_agg_V - np.maximum(0, np.max(-double_agg_psi, axis=1)/(1.-env.gamma))), 
        np.dot(env.rho, true_V), 
        np.dot(env.rho, true_V - np.maximum(0, np.max(-true_psi_t, axis=1)/(1.-env.gamma))),
    ))

    logger_validation.log(
        np.dot(env.rho, agg_V), 
        np.dot(env.rho, double_agg_V - np.maximum(0, np.max(-double_agg_psi, axis=1)/(1.-env.gamma))), 
        np.dot(env.rho, double_agg_V - np.max(-double_agg_psi)/(1.-env.gamma)),
        np.dot(env.rho, true_V), 
        np.dot(env.rho, true_V - np.maximum(0, np.max(-true_psi, axis=1)/(1.-env.gamma))),
        np.dot(env.rho, true_V - np.max(-true_psi)/(1.-env.gamma)),
        agg_V_err, 
        avg_total_V_err,
    )

    returned_f = np.dot(env.rho, true_V) if settings.get("tune_true_cost", False) else np.dot(env.rho, last_V_t)
    return pi_t, returned_f

def train(settings):
    seed_0 = settings["seed_0"]
    n_seeds = settings["n_seeds"]
    parallel = settings["parallel"]

    try:
        num_workers = min(n_seeds, len(os.sched_getaffinity(0))-1)
        print("Parallel PO experiements with %d workers (%d jobs, %d max cpu)" % (
            num_workers, n_seeds, mp.cpu_count())
        )
    except:
        if 'sched_getffinity' not in dir(os):
            print("Function `os.sched_getaffinity(0)` not available on current OS.\nSetting parallel=False.\nSee https://stackoverflow.com/questions/42658331/python-3-on-macos-how-to-set-process-affinity")
            parallel = False

    worker_queue = []

    for seed in range(seed_0, seed_0+n_seeds):
        customized_settings = settings.copy()
        customized_settings["seed"] = seed

        if not parallel:
            if customized_settings['tune_exploration']:
                _train_with_tuning(customized_settings)
            else:
                _train(customized_settings)
            continue

        if len(worker_queue) == num_workers:
            # wait for all workers to finish
            for p in worker_queue:
                p.join()
            worker_queue = []
        p = mp.Process(target=_train, args=(customized_settings,))
        p.start()
        worker_queue.append(p)

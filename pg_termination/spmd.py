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

def policy_update(pi, psi, eta):
    """ Closed-form solution with KL """
    (n_states, n_actions) = psi.shape

    assert (not np.any(np.isnan(psi))), "Found NaN in psi"
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

    (true_psi, true_V) = env.get_advantage(pi)

    for i in range(settings["validation_k"]):
        if settings["estimate_Q"] == "generative":
            (psi, V, _) = env.estimate_advantage_generative(pi, settings["N"], settings["T"])
        elif settings["estimate_Q"] == "online": # @depreciated
            (psi, V, _, _) = env.estimate_advantage_online_mc(pi, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_fixed":
            (psi, V, _, _) = env.estimate_advantage_online_mc(pi, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_estimate":
            t_mix, nu = env.get_mixing_time_ub(pi)
            (nu_est, tmix_est, _) = env.estimate_mixing_properties(pi, 0, tmix=t_mix, nu=nu)
            T = int(1./(1-env.gamma) + tmix_est/np.min(nu_est) + 1)
            (psi, V, _, _) = env.estimate_advantage_online_mc(pi, T, settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_dynamic":
            (psi, V, _, _) = env.estimate_advantage_online_mc_dynamic(pi, settings["pi_threshold"])
        elif settings["estimate_Q"] == "linear": # @depreciated
            (psi, V, _) = env.estimate_advantage_online_linear(pi, settings["T"])
        elif settings["estimate_Q"] == "ctd":
            # during validation, use online model
            (psi, V, _, _) = env.estimate_advantage_online_mc_dynamic(pi, settings["pi_threshold"])

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
        keys=["iter", "point value", "point opt_lb", "point uni_opt_lb", "agg value", "agg opt_lb", "agg uni_opt_lb", "true value", "true opt_lb", "true uni_opt_lb"], 
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

    log_min_T = np.log(settings['min_T'])/np.log(10)
    log_max_T = np.log(settings['max_T'])/np.log(10)
    Ts = np.power(10, rng.uniform(low=log_min_T, high=log_max_T, size=num_trials)).astype('int')
    tune_id_arr = np.arange(num_trials)
    tune_round = 0
    while 1:
        score_arr = np.zeros(len(Ts))
        Pi_arr = None
        # run all trials
        for i, T in enumerate(Ts):
            settings['T'] = T
            pi_t, f_t = _spmd(settings, 1.0, logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing)
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

def _spmd(settings, ukappa, logger, logger_agg_V, logger_agg_advgap, logger_validation, logger_mixing, pi_0=None):
    """
    SPMD training procedure 
    
    :param settings: dictionary of all user-defined parameter values
    :param ukappa: estimation of ukappa (if using CTD and not set, will set warning and default to 0)
    """
    assert ukappa > 0, "ukappa=%.4e not positive" % ukappa
    seed = settings['seed']

    # env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed, n_origins=5)
    env = mdpmodel.get_env(settings['env_name'], settings['gamma'], seed)

    if pi_0 is None:
        pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)
    next_greedy_pi_t = np.copy(pi_0)
    pi_star = None

    # copy of aggregate and true
    agg_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    agg_V_t = np.zeros(env.n_states, dtype=float)
    true_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    true_V_t = np.zeros(env.n_states, dtype=float)
    cum_samples = 0
    cum_est_samples = 0

    stepsize_scheduler = pmd.StepsizeSchedule(env, settings["stepsize_rule"], settings.get("eta",1))
    if "pi_threshold_mult" not in settings:
        warnings.warn("We changed 'pi_threshold'->'pi_threshold_mult'. Please update the yaml file (defaulting to 'pi_threshold_mult=1')")
        settings["pi_threshold_mult"] = 1.
    settings["pi_threshold"] = settings["pi_threshold_mult"] * (1.-env.gamma)/env.n_actions

    s_time = time.time()
    e_time = -time.time()
    if settings["estimate_Q"] == "linear": # @depreciated
        env.init_estimate_advantage_online_linear(settings)
    if settings["estimate_Q"] == "ctd": 
        Phi = env.rng.normal(size=(env.n_states*env.n_actions, settings["ctd_feature_size"]))
        theta = np.zeros(Phi.shape[1])

    for t in range(settings["n_iters"]):
        # this is only for logging/estimation
        n_est_samples = 0
        t_mix, nu = env.get_mixing_time_ub(pi_t)
        if not settings["skip_true_model"]:
            e_time += time.time() # skip advantage estimation time
            (true_psi_t, true_V_t) = env.get_advantage(pi_t)
            e_time -= time.time()

        if settings["estimate_Q"] == "generative":
            (psi_t, V_t, n_samples) = env.estimate_advantage_generative(pi_t, settings["N"], settings["T"])
        elif settings["estimate_Q"] == "online": # @depreciated
            (psi_t, V_t, _, n_samples) = env.estimate_advantage_online_mc(pi_t, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_fixed":
            (psi_t, V_t, _, n_samples) = env.estimate_advantage_online_mc(pi_t, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_estimate":
            (nu_est, tmix_est, n_est_samples) = env.estimate_mixing_properties(pi_t, 0, tmix=t_mix, nu=nu)
            T = int(1./(1-env.gamma) + tmix_est/np.min(nu_est) + 1)
            (psi_t, V_t, _, n_samples) = env.estimate_advantage_online_mc(pi_t, T, settings["pi_threshold"])
        elif settings["estimate_Q"] == "online_mc_dynamic":
            (psi_t, V_t, _, n_samples) = env.estimate_advantage_online_mc_dynamic(pi_t, settings["eps"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "linear": # @depreciated
            (psi_t, V_t, n_samples) = env.estimate_advantage_online_linear(pi_t, settings["T"])
        elif settings["estimate_Q"] == "ctd": 
            # pass in theta as last argument to warm start (doesn't help too much)
            (psi_t, V_t, n_samples, theta) = env.estimate_advantage_online_ctd(
                pi_t, Phi, ukappa, settings["eps"], settings["delta"], 
                settings['ctd_iota_mult'],
            )
        else: 
            raise Exception("Unknown estimate_Q setting %s" % settings["estimate_Q"])

        alpha_t = 1./(t+1)
        agg_psi_t = (1.-alpha_t)*agg_psi_t + alpha_t*psi_t
        agg_V_t = (1.-alpha_t)*agg_V_t + alpha_t*V_t
        cum_samples += n_samples
        cum_est_samples += n_est_samples

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
        logger_mixing.log(t, e_time + time.time(), cum_samples, cum_est_samples, np.min(nu), t_mix)

        eta_t = stepsize_scheduler.get_stepsize(t, psi_t)
        policy_update(pi_t, psi_t, eta_t) 

    print("Total runtime: %.2fs" % (time.time() - s_time))

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

    (true_psi, true_V) = env.get_advantage(pi_t)

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

    returned_f = np.dot(env.rho, true_V) if settings.get("tune_true_cost", False) else np.dot(env.rho, V_t)
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

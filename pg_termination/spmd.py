""" Basic stochastic PMD """
import time
import os
from enum import IntEnum
import multiprocessing as mp

import numpy as np

from pg_termination import pmd
from pg_termination import wbmdp 
from pg_termination import utils
from pg_termination.logger import BasicLogger

TOL = 1e-10

def policy_update(pi, psi, eta):
    """ Closed-form solution with KL """
    (n_states, n_actions) = psi.shape

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
            (psi, V) = env.estimate_advantage_generative(pi, settings["N"], settings["T"])
        elif settings["estimate_Q"] == "online":
            (psi, V, _) = env.estimate_advantage_online_mc(pi, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "linear":
            (psi, V) = env.estimate_advantage_online_linear(pi, settings["T"])

        agg_psi += psi
        agg_V += V
        total_V_err += np.max(np.abs(V - true_V))

    N = float(settings["validation_k"])
    agg_psi /= N
    agg_V /= N

    agg_V_err = np.max(np.abs(agg_V - true_V))
    avg_total_V_err = total_V_err / N

    print("Agg V err:   %.2e | Avg point V err: %.2e" % (agg_V_err, avg_total_V_err))

    return agg_psi, agg_V, agg_V_err, avg_total_V_err

def _train(settings):
    seed = settings['seed']

    # env = wbmdp.get_env(settings['env_name'], settings['gamma'], seed, n_origins=5)
    env = wbmdp.get_env(settings['env_name'], settings['gamma'], seed)

    logger = BasicLogger(
        fname=os.path.join(settings["log_folder"], "seed=%d.csv" % seed), 
        keys=["iter", "point value", "point opt_lb", "point uni_opt_lb", "agg value", "agg opt_lb", "agg uni_opt_lb", "true value", "true opt_lb", "true uni_opt_lb"], 
        dtypes=['d'] + ['f'] * 9
    )
    # logger_point_adv = BasicLogger(
    #     fname=os.path.join(settings["log_folder"], "point_agap_seed=%d.csv" % seed), 
    #     keys=["iter"] + ["(s_%d,a_%d)" % (s,a) for a in range(env.n_actions) for s in range(env.n_states)],
    #     dtypes=['d'] + ['f'] * (env.n_states*env.n_actions),
    # )
    # logger_agg_adv = BasicLogger(
    #     fname=os.path.join(settings["log_folder"], "agg_agap_seed=%d.csv" % seed), 
    #     keys=["iter"] + ["(s_%d,a_%d)" % (s,a) for a in range(env.n_actions) for s in range(env.n_states)],
    #     dtypes=['d'] + ['f'] * (env.n_states*env.n_actions),
    # )
    # logger_point_V = BasicLogger(
    #     fname=os.path.join(settings["log_folder"], "point_V_seed=%d.csv" % seed), 
    #     keys=["iter"] + ["s_%d" % s for s in range(env.n_states)],
    #     dtypes=['d'] + ['f'] * env.n_states,
    # )
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

    pi_0 = np.ones((env.n_actions, env.n_states), dtype=float)/env.n_actions
    pi_t = pi_0
    greedy_pi_t = np.copy(pi_0)
    next_greedy_pi_t = np.copy(pi_0)
    pi_star = None

    agg_psi_t = np.zeros((env.n_states, env.n_actions), dtype=float)
    agg_psi_t_halfway = np.zeros((env.n_states, env.n_actions), dtype=float) # only start counting halfway
    halfway_t = 5
    halfway_ct = 0
    agg_V_t = np.zeros(env.n_states, dtype=float)

    stepsize_scheduler = pmd.StepsizeSchedule(env, settings["stepsize_rule"], settings.get("eta",1))

    s_time = time.time()
    if settings["estimate_Q"] == "linear":
        env.init_estimate_advantage_online_linear(settings)

    for t in range(settings["n_iters"]):
        (true_psi_t, true_V_t) = env.get_advantage(pi_t)

        if settings["estimate_Q"] == "generative":
            (psi_t, V_t) = env.estimate_advantage_generative(pi_t, settings["N"], settings["T"])
        elif settings["estimate_Q"] == "online":
            (psi_t, V_t, _) = env.estimate_advantage_online_mc(pi_t, settings["T"], settings["pi_threshold"])
        elif settings["estimate_Q"] == "linear":
            (psi_t, V_t) = env.estimate_advantage_online_linear(pi_t, settings["T"])
        else: 
            raise Exception("Unknown estimate_Q setting %s" % settings["estimate_Q"])

        alpha_t = 1./(t+1)
        agg_psi_t = (1.-alpha_t)*agg_psi_t + alpha_t*psi_t
        agg_V_t = (1.-alpha_t)*agg_V_t + alpha_t*V_t
        if t >= halfway_t:
            alpha_t = 1./(t-halfway_t+1)
            agg_psi_t_halfway = (1-alpha_t)*agg_psi_t_halfway + alpha_t*psi_t
            halfway_ct += 1

        if ((t+1) <= 100 and (t+1) % 5 == 0) or (t+1) % 100==0:
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
            np.dot(env.rho, V_t - np.max(-psi_t, axis=1)/(1.-env.gamma)), 
            np.dot(env.rho, V_t - np.max(-psi_t)/(1.-env.gamma)), 
            np.dot(env.rho, agg_V_t), 
            np.dot(env.rho, agg_V_t - np.max(-agg_psi_t, axis=1)/(1.-env.gamma)), 
            np.dot(env.rho, agg_V_t - np.max(-agg_psi_t)/(1.-env.gamma)), 
            np.dot(env.rho, true_V_t), 
            np.dot(env.rho, true_V_t - np.max(-true_psi_t, axis=1)/(1.-env.gamma)),
            np.dot(env.rho, true_V_t - np.max(-true_psi_t)/(1.-env.gamma)),
        )
        # logger_point_adv.log(t+1, *list(psi_t.ravel()))
        # logger_agg_adv.log(t+1, *list(agg_psi_t.ravel()))
        # logger_point_V.log(t+1, *list(V_t))
        # logger_agg_V.log(t+1, *list(agg_V_t))

        eta_t = stepsize_scheduler.get_stepsize(t, psi_t)
        policy_update(pi_t, psi_t, eta_t) 

    print("Total runtime: %.2fs" % (time.time() - s_time))

    logger.save()
    # logger_point_adv.save()
    # logger_agg_adv.save()
    # logger_point_V.save()
    # logger_agg_V.save()

    # policy validation
    output = policy_validation(env, pi_t, settings)
    (agg_psi, agg_V, agg_V_err, avg_total_V_err) = output
    alpha = float(settings["validation_k"])/(settings["validation_k"] + halfway_ct)
    double_agg_psi = alpha*agg_psi + (1.-alpha)*agg_psi_t_halfway
    # double_agg_psi = agg_psi

    # logger_agg_adv.log(t+1, *list(double_agg_psi.ravel()))
    logger_agg_V.log(t+1, *list(agg_V))
    double_agg_advgap = np.max(-double_agg_psi, axis=1) 
    logger_agg_advgap.log(t+1, *list(double_agg_advgap))

    logger_agg_V.save()
    logger_agg_advgap.save()

    (true_psi, true_V) = env.get_advantage(pi_t)

    print("Offline: f=%.2e (fstar=%.2e) | true_f=%.2e (est_true_f_star=%.2e)" % (
        np.dot(env.rho, agg_V), 
        np.dot(env.rho, agg_V - np.max(-double_agg_psi, axis=1)/(1.-env.gamma)), 
        np.dot(env.rho, true_V), 
        np.dot(env.rho, true_V - np.max(-true_psi_t, axis=1)/(1.-env.gamma)),
    ))

    logger_validation.log(
        np.dot(env.rho, agg_V), 
        np.dot(env.rho, agg_V - np.max(-double_agg_psi, axis=1)/(1.-env.gamma)), 
        np.dot(env.rho, agg_V - np.max(-double_agg_psi)/(1.-env.gamma)),
        np.dot(env.rho, true_V), 
        np.dot(env.rho, true_V - np.max(-true_psi, axis=1)/(1.-env.gamma)),
        np.dot(env.rho, true_V - np.max(-true_psi)/(1.-env.gamma)),
        agg_V_err, 
        avg_total_V_err,
    )
    logger_validation.save()

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


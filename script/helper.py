import math
from pg_termination import pmd

def parse_sub_runs(sub_runs, total_runs):
    start_run_id, end_run_id = 0, total_runs
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= total_runs, "sub_runs id must be in [0,%s]" % (total_runs-1)
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % (total_runs-1))

    return start_run_id, end_run_id

def get_parameter_settings(seed_0, n_seeds, n_iters, print_info, about):
    od = dict([
        ("seed_0", seed_0), 
        ("n_seeds", n_seeds), 
        ("n_iters", n_iters),
        ("max_runtime_in_sec", 3600),
        ("min_obs", 0), 
        ("max_obs", math.inf),
        ("alg", "spmd"),
        ("eps", math.exp(-10)),
        ("delta", 1e-2),
        ("stepsize_rule", int(pmd.StepSize.SUBLINEAR)), 
        ("update_rule", int(pmd.Update.KL_UPDATE)),
        ("estimate_Q", "online_mc_fixed"),
        ("env_name", "discrete_mountaincar"), # change to gridworld_small
        ("gamma", 0.9),
        ("N_mc", 1), 
        ("T_mc", 1000), 
        ("validation_mode", None),
        ("validation_k", 0), 
        ("pi_threshold_mult", 1.0),
        ("eta", 0.01),
        ("skip_true_model", True),
        ("ctd_feature_size_ratio", 1.), 
        ("ctd_iota_mult", 40), # change to 1
        ("ctd_state_expl", False), 
        ("tune_exploration", False),
        ("successive_half_trials", 16), # change to 4
        ("min_T_mc", 100), # change to 1e2
        ("max_T_mc", 2e6), # change to 1e4
        ("qlearn_alpha", -1),
    ])

    od_info = [
        ("seed_0", "start seed"), 
        ("n_seeds", "num seeds"), 
        ("n_iters", "num SPMD iters"),
        ("max_runtime_in_sec", "max runtime before SPMD early terminates (only for SPMD)"),
        ("alg", "which algorithm to run ('pmd', 'spmd', 'policyiter')"),
        ("eps", "acc tolerance. Used for dynamic mixing time and CTD"),
        ("delta", "failure rate (only used by CTD for robust est)"),
        ("stepsize_rule", "stepsize rule (const, decr). See pmd.StepSizeint enum"),
        ("update_rule", "pmd update (euc, kl, tsallis)"),
        ("estimate_Q", "how to estimate Q (generative, online_mc_fixed, online_mc_estimate, online_mc_dynamic)"),
        ("env_name", "environment"),
        ("gamma", "discount factor"),
        ("N_mc", "number of replications in estimating Q (only for gen)"),
        ("T_mc", "Monte Carlo estimation length (only for gen, non-dynamic mc)"), 
        ("validation_k", "offline validation step replication amt"),
        ("pi_threshold_mult", "constant factor in cut-off for sub-opt actions"),
        ("eta", "base step size"),
        ("skip_true_model", "skips validation on true model - only works for non-gym envs"),
        ("ctd_feature_size_ratio", "feature size for CTD (only)"),
        ("ctd_iota_mult", "User chosen CTD stepsize multiplier"),
        ("ctd_state_expl", "Apply explicit state exploration [if func-approx unknown]"), 
        ("ctd_Phi_d", "Feature size in CTD. If -1, defaults to full size"),
        ("tune_exploration", "Tune exploration time in Monte Carlo"),
        ("successive_half_trials", "Number of trials in successive halving tuning"),
        ("min_T_mc", "Minimum Monte Carlo exploration time in tuning"),
        ("max_T_mc", "Maximum Monte Carlo exploration time in tuning"),
    ]

    if print_info:
        print("About:\n\t%s" % about)
        exp_metadata = ["setting", "description"]
        row_format ="{:<20}|{:<60}"
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (80+len(exp_metadata)-1))
        for name, description in od_info:
            print(row_format.format(name, description))
        print("-" * (80+len(exp_metadata)-1))

    return od

def get_clean_parameter_settings(seed_0, n_seeds, n_iters, print_info, about):
    od = dict([
        ("seed_0", seed_0), 
        ("n_seeds", n_seeds), 
        ("num_iterations", n_iters), 
        ("max_runtime_in_sec", 3600),
        ("alg", "clean_ppo"),
        ("env_name", "gridworld_small"), 
        ("gamma", 0.99),
        ("torch_deterministic", True),
        ("cuda", False),
        ("num_steps: 128"), # tune this
        ("total_timesteps", 500_000),
        ("learning_rate", 2.5e-4), # tune this
        ("anneal_lr", True),
        ("gae_lambda", 0.95),
        ("num_minibatches", 4),
        ("update_epochs", 4),
        ("norm_adv", True),
        ("clip_coef", 0.2),
        ("clip_vloss", True),
        ("ent_coef", 0.01),
        ("vf_coef", 0.5),
        ("max_grad_norm", 0.5),
        ("target_kl", None),
    ])

    # TODO: Update this
    od_info = [
        ("seed_0", "start seed"), 
        ("n_seeds", "num seeds"), 
        ("n_iters", "num SPMD iters"),
        ("max_runtime_in_sec", "max runtime before SPMD early terminates (only for SPMD)"),
        ("alg", "which algorithm to run ('pmd', 'spmd', 'policyiter')"),
        ("eps", "acc tolerance. Used for dynamic mixing time and CTD"),
        ("delta", "failure rate (only used by CTD for robust est)"),
        ("stepsize_rule", "stepsize rule (const, decr). See pmd.StepSizeint enum"),
        ("update_rule", "pmd update (euc, kl, tsallis)"),
        ("estimate_Q", "how to estimate Q (generative, online_mc_fixed, online_mc_estimate, online_mc_dynamic)"),
        ("env_name", "environment"),
        ("gamma", "discount factor"),
        ("N_mc", "number of replications in estimating Q (only for gen)"),
        ("T_mc", "Monte Carlo estimation length (only for gen, non-dynamic mc)"), 
        ("validation_k", "offline validation step replication amt"),
        ("pi_threshold_mult", "constant factor in cut-off for sub-opt actions"),
        ("eta", "base step size"),
        ("skip_true_model", "skips validation on true model - only works for non-gym envs"),
        ("ctd_feature_size", "feature size for CTD (only)"),
        ("ctd_iota_mult", "User chosen CTD stepsize multiplier"),
        ("tune_exploration", "Tune exploration time in Monte Carlo"),
        ("successive_half_trials", "Number of trials in successive halving tuning"),
        ("min_T_mc", "Minimum Monte Carlo exploration time in tuning"),
        ("max_T_mc", "Maximum Monte Carlo exploration time in tuning"),
    ]

    if print_info:
        print("About:\n\t%s" % about)
        exp_metadata = ["setting", "description"]
        row_format ="{:<20}|{:<60}"
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (80+len(exp_metadata)-1))
        for name, description in od_info:
            print(row_format.format(name, description))
        print("-" * (80+len(exp_metadata)-1))

    return od

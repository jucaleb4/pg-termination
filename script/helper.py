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
        ("log_folder", ""),
        ("seed_0", seed_0), 
        ("n_seeds", n_seeds), 
        ("n_iters", n_iters),
        ("max_runtime_in_sec", 3600),
        ("min_obs", 0), 
        ("max_obs", math.inf),
        ("alg", "spmd"),
        ("eps", math.exp(-10)),
        ("delta", 1e-2),
        ("save_policy", False),
        ("no_validation_gamma", False),
        ("stepsize_rule", int(pmd.StepSize.SUBLINEAR)), 
        ("update_rule", int(pmd.Update.KL_UPDATE)),
        ("estimate_Q", "online_mc_fixed"),
        ("env_name", "discrete_mountaincar"), # change to gridworld_small
        ("gamma", 0.9),
        ("N_mc", 1), 
        ("T_mc", 1000), 
        ("n_batches", 1),
        ("validation_mode", None),
        ("validation_k", 0), 
        ("pi_threshold_mult", 1.0),
        ("eta", 0.01),
        ("skip_true_model", True),
        ("ctd_feat_type", "Gaussian"),
        ("ctd_feat_size", None),
        ("ctd_feature_size_ratio", None),  # 0
        ("ctd_reg_val", None),
        ("ctd_reg_ratio", None), # 0
        ("ctd_iota_mult", None), # 1
        ("ctd_state_expl", None), # False
        ("ctd_burn_in", None), # False
        ("ctd_N_mult", 1.0), 
        ("ctd_uLam_mult", 1),
        ("ctd_ortho_feat", False),
        ("s_origin", 'reset'),
        ("tune_exploration", False),
        ("successive_half_trials", 16), # change to 4
        ("min_T_mc", 100), # change to 1e2
        ("max_T_mc", 2e6), # change to 1e4
        ("qlearn_alpha", -1),
        ("ukappa", 1),
        ("ppo_cuda", False),
        ("ppo_lr", 2.5e-4),
        ("ppo_rollout_len", 128),
        ("ppo_anneal_lr", True),
        ("ppo_gae_lambda", 0.95),
        ("ppo_num_minibatches", 1),
        ("ppo_update_epochs", 4),
        ("ppo_norm_adv", True),
        ("ppo_clip_coef", 0.2),
        ("ppo_clip_vloss", True),
        ("ppo_ent_coef", 0.01),
        ("ppo_vf_coef", 0.5),
        ("ppo_max_grad_norm", 0.5),
        ("ppo_target_kl", None),
    ])

    od_info = [
        ("log_folder", "Log folder"),
        ("seed_0", "start seed"), 
        ("n_seeds", "num seeds"), 
        ("n_iters", "num SPMD iters"),
        ("max_runtime_in_sec", "max runtime before SPMD early terminates (only for SPMD)"),
        ("alg", "which algorithm to run ('pmd', 'spmd', 'policyiter')"),
        ("eps", "acc tolerance. Used for dynamic mixing time and CTD"),
        ("delta", "failure rate (only used by CTD for robust est)"),
        ("save_policy", "Save last-iterate policy"),
        ("no_validation_gamma", "Remove discount factor from eval"),
        ("stepsize_rule", "stepsize rule (const, decr). See pmd.StepSizeint enum"),
        ("update_rule", "pmd update (euc, kl, tsallis)"),
        ("estimate_Q", "how to estimate Q (generative, online_mc_fixed, online_mc_estimate, online_mc_dynamic)"),
        ("env_name", "environment"),
        ("gamma", "discount factor"),
        ("N_mc", "number of replications in estimating Q (only for gen)"),
        ("T_mc", "Monte Carlo estimation length (only for gen, non-dynamic mc)"), 
        ("n_batches", "Number of mini-batches for policy eval (default: 1)"),
        ("validation_k", "offline validation step replication amt"),
        ("pi_threshold_mult", "constant factor in cut-off for sub-opt actions"),
        ("eta", "base step size"),
        ("skip_true_model", "skips validation on true model - only works for non-gym envs"),
        ("ctd_feat_type", "Feature type. Should be 'Gaussian' or 'rff'"),
        ("ctd_feat_size", "Feature size. If -1, full size"),
        ("ctd_feature_size_ratio", "feature size for CTD (only)"),
        ("ctd_reg_val", "regularization strength"),
        ("ctd_reg_ratio", "relative (to sig vals) regularization added to features"),
        ("ctd_iota_mult", "User chosen CTD stepsize multiplier"),
        ("ctd_state_expl", "Apply explicit state exploration [if func-approx unknown]"), 
        ("ctd_burn_in", "Burn in for CTD operator to use all discounted sums"),
        ("ctd_N_mult", "CTD iterations multiplier"),
        ("ctd_uLam_mult", "CTD parameter uLam multiplier"),
        ("ctd_ortho_feat", "Orthogonalize feature matrix. Does it automatically if d < |Z|"),
        ("s_origin", "Origin rule for CTD. None for fixed trajectory length, otherwise 'reset' or 'rand'"),
        ("tune_exploration", "Tune exploration time in Monte Carlo"),
        ("successive_half_trials", "Number of trials in successive halving tuning"),
        ("min_T_mc", "Minimum Monte Carlo exploration time in tuning"),
        ("max_T_mc", "Maximum Monte Carlo exploration time in tuning"),
        ("ukappa", "Lower bound on kappa"),
        ("ppo_cuda", "If toggled, cuda will be enabled by default"),
        ("ppo_lr", "the learning rate of the optimizer"),
        ("ppo_rollout_len", "the number of steps to run in each environment per policy rollout"),
        ("ppo_anneal_lr", "Toggle learning rate annealing for policy and value networks"),
        ("ppo_gae_lambda", "the lambda for the general advantage estimation"),
        ("ppo_num_minibatches", "the number of mini-batches"),
        ("ppo_update_epochs", "the K epochs to update the policy"),
        ("ppo_norm_adv", "Toggles advantages normalization"),
        ("ppo_clip_coef", "the surrogate clipping coefficient"),
        ("ppo_clip_vloss", "Toggles whether or not to use a clipped loss for the value function."),
        ("ppo_ent_coef", "coefficient of the entropy"),
        ("ppo_vf_coef", "coefficient of the value function"),
        ("ppo_max_grad_norm", "the maximum norm for the gradient clipping"),
        ("ppo_target_kl", "target_kl"),
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
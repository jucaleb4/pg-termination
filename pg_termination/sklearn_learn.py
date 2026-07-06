def init_estimate_advantage_online_linear(self, linear_settings):
    """ 
    Prepares radial basis functions for linear function approximation:

        https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html

    See also: https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb

    :param X: Nxn array of inputs, where N is the number of datapoints and n is the size of the state space
    """

    self.featurizer = sklearn.pipeline.FeatureUnion([
        # ("rbf0", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf1", sklearn.kernel_approximation.RBFSampler(gamma=1.0, n_components=100)),
        # ("rbf2", RBFSampler(gamma=0.1, n_components=100)),
    ])

    X = np.vstack((
        np.kron(np.arange(self.n_states), np.ones(self.n_actions)),
        np.kron(np.ones(self.n_states), np.arange(self.n_actions)),
    )).T

    self.featurizer.fit(X)
    self.model = sklearn.linear_model.SGDRegressor(
        learning_rate=linear_settings["linear_learning_rate"],
        eta0=linear_settings["linear_eta0"],
        max_iter=linear_settings["linear_max_iter"],
        alpha=linear_settings["linear_alpha"],
        warm_start=True, 
        tol=0.0,
        n_iter_no_change=linear_settings["linear_max_iter"],
        fit_intercept=True,
    )

    # We need to call partial_fit once to initialize the model or we get a
    # NotFittedError when trying to make a prediction This is quite hacky.
    self.model.partial_fit(self.featurize([X[0]]), [0])
    self.init_linear = True

def featurize(self, X):
    return self.featurizer.transform(X).astype('float64')

def predict(self, x):
    features = self.featurize(x)
    output = np.squeeze(self.model.predict(features))
    return output

def get_all_sa_pairs_for_finite(self):
    X_all_sa = np.vstack((
        np.kron(np.arange(self.n_states), np.ones(self.n_actions)),
        np.kron(np.ones(self.n_states), np.arange(self.n_actions)),
    )).T
    return X_all_sa

def custom_SGD(solver, X, y, minibatch=32):
    n_epochs = solver.max_iter
    n_consec_regress_epochs = 0
    max_regress = solver.n_iter_no_change
    frac_validation = solver.validation_fraction
    tol = solver.tol
    early_stopping = solver.early_stopping

    train_losses = []
    test_losses = []

    for i in range(n_epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, shuffle=True, test_size=frac_validation)
        num_batches = int(np.ceil(len(X_train)/ minibatch))
        for j in range(num_batches):
            k_s = minibatch*j
            k_e = min(len(X_train), minibatch*(j+1))
            # mini-batch update
            solver.partial_fit(X_train[k_s:k_e], y_train[k_s:k_e])

        y_train_pred = solver.predict(X_train)
        y_test_pred = solver.predict(X_test)

        train_losses.append(la.norm(y_train_pred - y_train)**2/len(y_train))
        test_losses.append(la.norm(y_test_pred - y_test)**2/len(y_test))

        if early_stopping and len(test_losses) > 1 and test_losses[-1] > np.min(test_losses)-tol:
            n_consec_regress_epochs += 1
        else:
            n_consec_regress_epochs = 0
        if n_consec_regress_epochs == max_regress:
            print("Early stopping (stagnate)")
            break
        if train_losses[-1] <= tol:
            print("Early stopping (train loss small)")
            break

    return np.array(train_losses), np.array(test_losses)


# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_early_stopping.html#sphx-glr-auto-examples-linear-model-plot-sgd-early-stopping-py
@ignore_warnings(category=ConvergenceWarning)
def estimate_advantage_online_linear(self, pi, T):
    """
    Use Monte Carlo simulation to obtain partial Q function.  We use linear
    function approximation with bootstrap to update sampled sa pairs and
    fill in missing sa pairs.

    :param T: duration to run Monte Carlo simulation
    """
    assert self.init_linear, "Run `init_estimate_advantage_online_linear` before estimating"

    # use monte carlo estimate to estimate truncated psi (threshold=0
    # ensures non-visited sa have zero value, i.e., Q[s,a]=0)
    output = self.estimate_advantage_online_mc(pi, T, threshold=0, bootstrap=True)
    (psi, V_pi, visit_len_state_action) = output
    Q = psi + np.outer(V_pi, np.ones(self.n_actions, dtype=float))

    # bootstrap remaining cost-to-go values
    # X_all_sa = self.get_all_sa_pairs_for_finite()
    # y = self.predict(X_all_sa)

    visited_sa_s, visited_sa_a = np.where(visit_len_state_action >= 1)
    X_visited_sa = np.vstack((visited_sa_s, visited_sa_a)).T
    # state-action pair index in 1D
    visited_idxs = self.n_actions * visited_sa_s + visited_sa_a

    # y = Q.flatten() + np.multiply(np.power(self.gamma, visit_len_state_action.flatten()), y)
    # visited_idxs = np.where(visit_len_state_action.flatten() > 0)[0]
    y = Q.flatten()[visited_idxs]
    # for i, (s,a) in zip(visited_idxs, X_visited_sa):
    #     y[i] = Q[s,a] + self.gamma**visit_len_state_action[s,a]*y[i]

    # training update
    # features = self.featurize(X_visited_sa)
    # self.model.fit(features, y[visited_idxs])
    # features = self.featurize(X_all_sa)
    # self.model.fit(features, y)
    features = self.featurize(X_visited_sa)
    self.model.fit(features, y)

    # predict psi_pi
    X_all_sa = self.get_all_sa_pairs_for_finite()
    q_pred = self.predict(X_all_sa)
    Q_pred = np.reshape(q_pred, newshape=(self.n_states, self.n_actions))
    V_pred = np.einsum('sa,as->s', Q_pred, pi)
    psi_pred = Q_pred - np.outer(V_pred, np.ones(self.n_actions, dtype=float))

    return (psi_pred, V_pred)

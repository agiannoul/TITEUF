import numpy as np

def sigmoid_survival(times, rhat, tau=1):
    """Smooth survival using a logistic (sigmoid) CDF centered at rhat.
    tau controls the transition width (larger -> smoother)."""
    times = np.array(times)
    return 1.0 / (1.0 + np.exp((times - rhat) / tau))  # ~1 for t<<rhat, ~0 for t>>rhat

def sigmoid_survival_batch(times, rhat_batch, tau=1):
    all_survivals = []
    for rhat in rhat_batch:
        survival = sigmoid_survival(times, rhat, tau)
        all_survivals.append(survival)
    return np.array(all_survivals)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=0)

def softmax_distance_survival(times, rhat, tau=1, kernel='laplace'):
    """Create a discrete failure-time pmf using softmax over distances to rhat,
    then convert to survival S(t)=P(T>t) = 1 - CDF(t).

    kernel: 'laplace' uses exp(-|t-r|/tau) (like softmax on absolute distance)
            'gaussian' uses exp(-(t-r)^2/(2*tau^2))
    """
    times = np.array(times)
    if kernel == 'laplace':
        logits = -np.abs(times - rhat) / tau
    elif kernel == 'gaussian':
        logits = -((times - rhat )**2) / (2.0 * (tau**2))
    else:
        raise ValueError("kernel must be 'laplace' or 'gaussian'")
    pmf = softmax(logits)
    # compute survival: S(t_k) = sum_{j: t_j > t_k} pmf_j
    cdf = np.cumsum(pmf)
    survival = 1.0 - cdf + pmf  # survival at t_k should include probability at exactly t_k? adjust if desired
    # The line above gives S(t_k)=sum_{j>=k+1} pmf_j + pmf_k; depends on your time grid convention.
    # A clearer approach below: survival at t is P(T>t) -> sum_{j: times[j] > t} pmf_j
    survival_strict = np.array([pmf[times > t_k].sum() for t_k in times])
    return pmf, survival_strict


def softmax_distance_survival_batch(times, rhat_batch, tau=1, kernel='laplace'):
    """Batch version of softmax_distance_survival for multiple rhat predictions."""
    # all_pmfs = []
    all_survivals = []
    for rhat in rhat_batch:
        _, survival = softmax_distance_survival(times, rhat, tau, kernel)
        # all_pmfs.append(pmf)
        all_survivals.append(survival)
    return np.array(all_survivals)

def hard_transform_survival(times, flatten_preds):
    """Hard step survival: S(t)=1 for t<rhat, S(t)=0 for t>=rhat."""
    test_preds = []
    for pred in flatten_preds:
        surv_pred = []
        for t in times:
            if t < pred:
                surv_pred.append(1.0)
            else:
                surv_pred.append(0.0)
        test_preds.append(surv_pred)
    test_preds = np.array(test_preds)
    return test_preds
import numpy as np
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

def p_adjust_fdr(p_vals):
    # Benjamini-Hochberg FDR control
    rej, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
    return pvals_corrected

def sequential_BH(true_stat, perm_stat, M, B, peeking_times, alpha=0.05, param_bm=0.9, param_bc=10,
                  param_amt=0.01, peeking_times_AMT=None, method="bm"):
    """
    Applies Benjamini Hochberg procedure to sequential permutation p-values
    """
    if peeking_times_AMT is None:
        # Default peeking times for AMT as in R: floor(c(cumsum(c(B / 100, B / 100 * cumprod(rep(1.1, 24)))), B))
        pt = [B / 100]
        for _ in range(24):
            pt.append(pt[-1] * 1.1)
        pt = np.cumsum(pt).tolist()
        pt.append(B)
        peeking_times_AMT = np.floor(pt).astype(int)

    if method == "bm":
        c_param = param_bm * alpha
        rejects = []
        consider = list(range(M))
        losses = np.zeros(M, dtype=int)
        lowest_rej = np.full(M, M + 1)
        m_star = 1
        stop = np.full(M, B)

        for i in peeking_times:
            # Calculate critical vector for number of losses
            # R: qbinom(1 - param_bm, i + 1, param_bm * alpha * (1:M) / M) - 1
            # Python binom.ppf corresponds to qbinom
            probs = param_bm * alpha * np.arange(1, M + 1) / M
            critical = binom.ppf(1 - param_bm, i + 1, probs) - 1
            
            # Calculate number of losses
            # perm_stat is shape (M, B). Indexing up to i
            # In R: perm_stat[consider, (1:i)] >= true_stat[consider]
            if len(consider) > 0:
                if i > 1:
                    losses[consider] = np.sum(perm_stat[np.ix_(consider, range(i))] >= true_stat[consider, None], axis=1)
                else:
                    losses[consider] = (perm_stat[consider, 0] >= true_stat[consider]).astype(int)

            max_crit = np.max(critical) if len(critical) > 0 else -1

            if max_crit >= 0:
                # In R: idx_critical <- match((0:max_crit), critical)
                # match returns the first 1-based index of the first match.
                # In python, we find first occurrence of each value in 0..max_crit
                idx_critical = []
                for val in range(int(max_crit) + 1):
                    # find first index where critical == val
                    idxs = np.where(critical == val)[0]
                    if len(idxs) > 0:
                        idx_critical.append(idxs[0] + 1) # 1-based index to match M
                
                idx_critical.append(M + 1)
                idx_critical = np.array(idx_critical)
                
                # Calculate lowest level a p-value could reject at
                # R: pmin(lowest_rej[consider], idx_critical[length(idx_critical) - rowSums(outer((losses[consider]), critical, "<="))])
                # We count how many elements in `critical` are >= `losses[consider]`
                # sum(losses <= critical)
                for u in consider:
                    num_less_equal = np.sum(losses[u] <= critical)
                    val_idx = len(idx_critical) - num_less_equal - 1
                    # In python it's 0-based for the list idx_critical, but R's length(idx_critical) - sum is 1-based logic?
                    # Let's be careful. If sum = 0, we want the last element (M+1). So index is len(idx_critical) - 1.
                    val = idx_critical[val_idx]
                    lowest_rej[u] = min(lowest_rej[u], val)
                
                # Calculate m_star
                sort_lowest_rej = np.sort(lowest_rej)
                # which(sort_lowest_rej <= (1:M))
                valid_m = np.where(sort_lowest_rej <= np.arange(1, M + 1))[0]
                if len(valid_m) > 0:
                    m_star = valid_m[-1] + 1 # 1-based
                else:
                    m_star = 1
                
                # Set rejections and stops
                new_rejects = np.where(lowest_rej <= m_star)[0].tolist()
                rejects = new_rejects
                consider = list(set(consider) - set(rejects))

            # Stop for futility
            # R: (1 - pbinom(losses, i + 1, param_bm * alpha * (length(consider) + m_star) / M)) / (...) < alpha * ...
            if len(consider) > 0:
                prob = param_bm * alpha * (len(consider) + m_star) / M
                # binom.cdf(k, n, p) -> P(X <= k). 1 - P(X <= k-1) = P(X >= k)
                # But R's pbinom(q, size, prob) is P(X <= q).
                # 1 - pbinom(losses, i + 1, prob)
                num = 1 - binom.cdf(losses[consider], i + 1, prob)
                den = prob
                thresh = alpha * (len(consider) + m_star) / M
                
                futility_cond = (num / den) >= thresh # We keep if >= thresh, so we stop if < thresh
                
                keep = []
                stopped = []
                for idx, c_idx in enumerate(consider):
                    if futility_cond[idx]:
                        keep.append(c_idx)
                    else:
                        stopped.append(c_idx)
                
                consider = keep
                
                # update stop
                not_consider = list(set(range(M)) - set(consider))
                stop[not_consider] = np.minimum(i, stop[not_consider])

            if len(consider) == 0:
                break
                
        return {"rejects": rejects, "stop": stop}

    elif method == "bc":
        h = param_bc
        rejects = []
        consider = list(range(M))
        losses = np.zeros(M, dtype=int)
        lowest_rej = np.full(M, M + 1)
        stop = np.full(M, B)

        for i in peeking_times:
            # R: pmin(floor((i + h - h / (alpha * (1:M) / M))), h - 1)
            critical = np.minimum(np.floor(i + h - h / (alpha * np.arange(1, M + 1) / M)), h - 1)
            
            if len(consider) > 0:
                if i > 1:
                    losses[consider] = np.sum(perm_stat[np.ix_(consider, range(i))] >= true_stat[consider, None], axis=1)
                else:
                    losses[consider] = (perm_stat[consider, 0] >= true_stat[consider]).astype(int)

            max_crit = np.max(critical) if len(critical) > 0 else -1

            if max_crit >= 0:
                idx_critical = []
                for val in range(int(max_crit) + 1):
                    idxs = np.where(critical == val)[0]
                    if len(idxs) > 0:
                        idx_critical.append(idxs[0] + 1)
                
                idx_critical.append(M + 1)
                idx_critical = np.array(idx_critical)
                
                for u in consider:
                    num_less_equal = np.sum(losses[u] <= critical)
                    val_idx = len(idx_critical) - num_less_equal - 1
                    lowest_rej[u] = min(lowest_rej[u], idx_critical[val_idx])
                
                sort_lowest_rej = np.sort(lowest_rej)
                valid_m = np.where(sort_lowest_rej <= np.arange(1, M + 1))[0]
                m_star = valid_m[-1] + 1 if len(valid_m) > 0 else 1
                
                rejects = np.where(lowest_rej <= m_star)[0].tolist()
                consider = list(set(consider) - set(rejects))

            # Stop for futility: losses >= h
            if len(consider) > 0:
                keep = []
                for c_idx in consider:
                    if losses[c_idx] < h:
                        keep.append(c_idx)
                
                consider = keep
                not_consider = list(set(range(M)) - set(consider))
                stop[not_consider] = np.minimum(i, stop[not_consider])

            if len(consider) == 0:
                break
                
            if i == max(peeking_times):
                p_vals = np.ones(M)
                if len(rejects) > 0:
                    p_vals[rejects] = 0
                if len(consider) > 0:
                    p_vals[consider] = (losses[consider] + 1) / (i + 1)
                
                rej_mask, _, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')
                rejects = np.where(rej_mask)[0].tolist()
                
        return {"rejects": rejects, "stop": stop}

    elif method == "AMT":
        rejects = []
        consider = list(range(M))
        losses = np.zeros(M, dtype=int)
        r_hat = alpha
        p_lower = np.zeros(M)
        p_upper = np.ones(M)
        time = np.zeros(M, dtype=int) # index in peeking_times_AMT (0-based)
        
        # We need binom_confint equivalent
        from statsmodels.stats.proportion import proportion_confint
        
        while True:
            time[consider] += 1
            
            for u in consider:
                t_idx = time[u] - 1
                curr_peeking_time = peeking_times_AMT[t_idx]
                losses[u] = np.sum(perm_stat[u, :curr_peeking_time] >= true_stat[u])
                
                if time[u] < len(peeking_times_AMT):
                    conf_level = 1 - (param_amt / (2 * M * len(peeking_times_AMT)))
                    alpha_conf = 1 - conf_level
                    # Agresti-Coull method is 'agresti_coull' in statsmodels
                    ci_low, ci_upp = proportion_confint(losses[u], curr_peeking_time, alpha=alpha_conf, method='agresti_coull')
                    p_lower[u] = ci_low
                    p_upper[u] = ci_upp
                else:
                    val = (losses[u] + 1) / (B + 1)
                    p_lower[u] = val
                    p_upper[u] = val
                    
            for j in range(M, -1, -1):
                r_hat = j
                C_g = np.where(p_lower > (j / M * alpha))[0]
                if r_hat == (M - len(C_g)):
                    break
                    
            tau_hat = r_hat * alpha / M
            
            consider = np.where((p_lower <= tau_hat) & (p_upper > tau_hat))[0].tolist()
            rejects = np.where(p_upper <= tau_hat)[0].tolist()
            
            if len(consider) == 0:
                break
                
        # The peeking_times_AMT array in R is accessed with 1-based index, we use 0-based time-1
        stop = np.array([peeking_times_AMT[min(t - 1, len(peeking_times_AMT) - 1)] for t in time])
        return {"rejects": rejects, "stop": stop}

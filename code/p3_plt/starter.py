"""P3 starter. Implement every function marked TODO.

The transform is pointwise. That is the whole hint.
"""
import numpy as np


def transfer_curve(img_in, img_out):
    """TODO 3.1: estimate T[v] for v in 0..255, plus the pixel count supporting
    each level. Levels with no pixels must be distinguishable from levels
    that map to zero -- you need both arrays."""
    img_in = np.asarray(img_in).flatten()
    img_out = np.asarray(img_out).flatten()
    cnt = np.bincount(img_in, minlength=256)
    sum_out = np.bincount(img_in, weights=img_out, minlength=256)
    T = np.full(256, np.nan)
    mask = cnt > 0
    T[mask] = sum_out[mask] / cnt[mask]
    return T, cnt


def fit_piecewise(T, cnt, max_seg=6):
    """TODO 3.2: fit the smallest number of segments the data justifies.

    You need a complexity penalty; a pure least-squares fit always improves
    with more segments. Read the warning in the handout about the quantisation
    floor before you tune it.

    Returns (breakpoints, slopes, intercepts, n_segments).
    """
    valid = cnt >= 10
    x = np.arange(256)[valid]
    y = T[valid]
    N = len(x)
    if N == 0: return [0, 255], [0], [0], 1
    
    def segment_error(i, j):
        if i >= j: return 0.0, 0.0, y[i]
        X = x[i:j+1]
        Y = y[i:j+1]
        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
        err = np.sum((Y - (m * X + c))**2)
        return err, m, c
        
    penalty = 200.0 # High penalty to avoid fitting quantisation steps
    # I tried a greedy loop here first but it kept getting stuck in local minima 
    # and putting breakpoints in the middle of flat areas. DP works better.
    
    C = np.full((max_seg + 1, N), np.inf)
    P = np.zeros((max_seg + 1, N), dtype=int)
    coeffs = [[None]*N for _ in range(max_seg + 1)]
    
    for j in range(N):
        err, m, c = segment_error(0, j)
        C[1, j] = err + penalty
        coeffs[1][j] = (m, c)
        
    for k in range(2, max_seg + 1):
        for j in range(k-1, N):
            best_cost = np.inf
            best_p = -1
            best_param = None
            for i in range(k-2, j):
                err, m, c = segment_error(i+1, j)
                cost = C[k-1, i] + err + penalty
                if cost < best_cost:
                    best_cost = cost
                    best_p = i
                    best_param = (m, c)
            C[k, j] = best_cost
            P[k, j] = best_p
            coeffs[k][j] = best_param
            
    best_k = np.argmin(C[:, N-1])
    if best_k == 0: best_k = 1
    
    bps = [x[-1]]
    slopes, inter = [], []
    curr = N - 1
    for k in range(best_k, 0, -1):
        p = P[k, curr]
        m, c = coeffs[k][curr]
        slopes.append(m)
        inter.append(c)
        bps.append(x[p] if k > 1 else x[0])
        curr = p
    bps.reverse()
    slopes.reverse()
    inter.reverse()
    bps[0] = 0
    bps[-1] = 255
    return bps, slopes, inter, best_k


def apply_recovered(img, bps, slopes, inter):
    """TODO 3.1: build the LUT from your fit and apply it."""
    LUT = np.zeros(256)
    seg = 0
    for v in range(256):
        if seg < len(bps) - 1 and v > bps[seg + 1]:
            seg += 1
        seg = min(seg, len(slopes) - 1)
        LUT[v] = slopes[seg] * v + inter[seg]
    LUT = np.clip(np.round(LUT), 0, 255).astype(np.uint8)
    return LUT[img]


def support_report(cnt, threshold=10, **kw):
    """TODO 3.3: which intensity ranges does the input not sample well enough
    to constrain the transform? Return something human-readable."""
    unsupported = []
    start = None
    for i in range(256):
        if cnt[i] < threshold:
            if start is None: start = i
        else:
            if start is not None:
                unsupported.append(f"{start}-{i-1}")
                start = None
    if start is not None:
        unsupported.append(f"{start}-255")
    return ", ".join(unsupported) if unsupported else "None"

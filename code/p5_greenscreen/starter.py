"""P5 starter. Implement every function marked TODO."""
import numpy as np


def to_ycbcr(rgb):
    """TODO 5.1: BT.601 conversion."""
    M = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.5     ],
        [ 0.5,      -0.418688, -0.081312]
    ])
    ycbcr = np.dot(rgb.astype(float), M.T)
    ycbcr[:,:,1:] += 128
    return np.clip(ycbcr, 0, 255)


def key_naive_rgb(rgb, thresh=60):
    """TODO 5.1: green dominance in raw RGB. Expected to fail on the unevenly
    lit backing -- show where."""
    R, G, B = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    mask = (G > R + thresh) & (G > B + thresh)
    return (~mask).astype(np.float64)


def estimate_key_colour(rgb, border=24):
    """TODO 5.2: estimate the backing chroma from the frame. Do not hard-code
    a green value; the graded footage varies."""
    ycbcr = to_ycbcr(rgb)
    top = ycbcr[:border, :, :]
    bottom = ycbcr[-border:, :, :]
    left = ycbcr[border:-border, :border, :]
    right = ycbcr[border:-border, -border:, :]
    
    border_pixels = np.concatenate([
        top.reshape(-1, 3), 
        bottom.reshape(-1, 3), 
        left.reshape(-1, 3), 
        right.reshape(-1, 3)
    ], axis=0)
    
    median_cb = np.median(border_pixels[:, 1])
    median_cr = np.median(border_pixels[:, 2])
    return np.array([median_cb, median_cr])


def key_soft(rgb, key_cbcr=None, t_in=None, t_out=None):
    """TODO 5.2: fractional alpha from chroma distance."""
    if key_cbcr is None:
        key_cbcr = estimate_key_colour(rgb)
    if t_in is None: t_in = 10.0
    if t_out is None: t_out = 30.0
        
    ycbcr = to_ycbcr(rgb)
    cbcr = ycbcr[:, :, 1:]
    
    dist = np.linalg.norm(cbcr - key_cbcr, axis=2)
    alpha = (dist - t_in) / (t_out - t_in)
    return np.clip(alpha, 0, 1)


def suppress_spill(rgb, alpha, strength=1.0):
    """TODO 5.2: remove green bounce. Should scale with (1 - alpha). Why?"""
    R = rgb[:, :, 0].astype(float)
    G = rgb[:, :, 1].astype(float)
    B = rgb[:, :, 2].astype(float)
    
    max_g = (R + B) * strength
    G_despill = np.where(G > max_g, max_g, G)
    G_final = G_despill * (1 - alpha) + G * alpha
    
    out = np.stack([R, G_final, B], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def composite(fg, bg, alpha, spill=True):
    """TODO 5.2: I = a*F + (1-a)*B."""
    if spill:
        fg = suppress_spill(fg, alpha)
    alpha_3 = alpha[:, :, None]
    out = fg.astype(float) * alpha_3 + bg.astype(float) * (1 - alpha_3)
    return np.clip(out, 0, 255).astype(np.uint8)


class ChromaLUT:
    """TODO 5.3: quantise (Cb,Cr) so keying is one table lookup per pixel."""

    def __init__(self, key_cbcr, t_in=None, t_out=None, bins=256):
        if t_in is None: t_in = 10.0
        if t_out is None: t_out = 30.0
        
        self.bins = bins
        self.lut = np.zeros((bins, bins), dtype=np.float32)
        
        for cb in range(bins):
            for cr in range(bins):
                dist = np.sqrt((cb - key_cbcr[0])**2 + (cr - key_cbcr[1])**2)
                alpha = (dist - t_in) / (t_out - t_in)
                self.lut[cb, cr] = np.clip(alpha, 0, 1)

    def alpha(self, rgb):
        rgb_f = rgb.astype(np.float32)
        Cb = np.clip(-0.168736 * rgb_f[:,:,0] - 0.331264 * rgb_f[:,:,1] + 0.5 * rgb_f[:,:,2] + 128, 0, 255).astype(int)
        Cr = np.clip(0.5 * rgb_f[:,:,0] - 0.418688 * rgb_f[:,:,1] - 0.081312 * rgb_f[:,:,2] + 128, 0, 255).astype(int)
        return self.lut[Cb, Cr]


# ---- provided metrics: do not modify
def sad(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / 1000.0)


def mse_alpha(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64))**2))


def grad_error(a, b):
    from scipy.ndimage import gaussian_filter
    ga = np.hypot(*np.gradient(gaussian_filter(a.astype(np.float64), 1.0)))
    gb = np.hypot(*np.gradient(gaussian_filter(b.astype(np.float64), 1.0)))
    return float(np.abs(ga - gb).sum() / 1000.0)


def temporal_flicker(alphas):
    if len(alphas) < 2:
        return 0.0
    total = 0.0
    count = 0
    prev = np.asarray(alphas[0], dtype=np.float32)
    for alpha in alphas[1:]:
        cur = np.asarray(alpha, dtype=np.float32)
        total += float(np.abs(cur - prev).sum(dtype=np.float64))
        count += cur.size
        prev = cur
    return total / count

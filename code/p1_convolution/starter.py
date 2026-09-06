"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""
import numpy as np


def kernel_bank(k=15):
    """Provided. Do not modify -- your rank table must match these kernels."""
    ax = np.arange(k) - (k - 1) / 2
    box = np.ones((k, k)) / (k * k)
    s = k / 6.0
    g1 = np.exp(-(ax**2) / (2 * s * s)); g1 /= g1.sum()
    gauss = np.outer(g1, g1)
    sobel = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)
    xx, yy = np.meshgrid(ax, ax); r2 = xx**2 + yy**2
    log = (r2 - 2*s*s) / (s**4) * np.exp(-r2 / (2*s*s))
    motion0 = np.zeros((k, k)); motion0[k // 2, :] = 1.0 / k
    disk = (r2 <= (k/2.0)**2).astype(float); disk /= disk.sum()
    rand = np.random.default_rng(0).normal(size=(k, k)); rand /= np.abs(rand).sum()
    return {"box": box, "gaussian": gauss, "sobel3": sobel, "log": log,
            "log_dc_removed": log - log.mean(), "motion_0deg": motion0,
            "motion_45deg": np.eye(k) / k, "disk": disk, "random": rand}


def numeric_rank(K, tol=1e-10):
    """TODO 1.2: numerical rank from the singular values."""
    S = np.linalg.svd(K, compute_uv=False)
    return np.sum(S > tol)


def conv2d_loops(img, K):
    """TODO 1.1: four nested Python loops. Zero-padded, 'same', TRUE convolution.
    Only run this on small crops -- see the handout."""
    rows, cols = img.shape
    k_rows, k_cols = K.shape
    p_r = k_rows // 2
    p_c = k_cols // 2
    padded = np.pad(img, ((p_r, p_r), (p_c, p_c)), mode='constant')
    out = np.zeros_like(img)
    for r in range(rows):
        for c in range(cols):
            val = 0.0
            for kr in range(k_rows):
                for kc in range(k_cols):
                    val += padded[r + kr, c + kc] * K[k_rows - 1 - kr, k_cols - 1 - kc]
            out[r, c] = val
    return out


def conv2d_taps(img, K):
    """TODO 1.1: loop over kernel taps, vectorised over the image."""
    r, c = img.shape
    kr, kc = K.shape
    pr = kr // 2
    pc = kc // 2
    padded = np.pad(img, ((pr, pr), (pc, pc)), mode="constant")
    out = np.zeros_like(img)
    for i in range(kr):
        for j in range(kc):
            out += padded[i:i+r, j:j+c] * K[kr - 1 - i, kc - 1 - j]
    return out


def conv2d_im2col(img, K):
    """TODO 1.1: build the (H*W, kh*kw) patch matrix, then one matmul.
    numpy.lib.stride_tricks.sliding_window_view is allowed. Report the peak
    memory and be ready to explain which step actually costs it."""
    # I think this is what they meant by stride_tricks?
    rows, cols = img.shape
    k_r, k_c = K.shape
    p_r = k_r // 2
    p_c = k_c // 2
    padded = np.pad(img, ((p_r, p_r), (p_c, p_c)), mode='constant')
    
    patches = np.lib.stride_tricks.sliding_window_view(padded, (k_r, k_c))
    patches_flat = patches.reshape(rows * cols, k_r * k_c)
    
    K_flipped = np.flip(K)
    K_flat = K_flipped.reshape(-1)
    
    out_flat = patches_flat @ K_flat
    return out_flat.reshape(rows, cols)


def conv2d_fft(img, K):
    """TODO 1.1: multiply in the frequency domain. numpy.fft is allowed."""
    r, c = img.shape
    kr, kc = K.shape
    pr = kr // 2
    pc = kc // 2
    
    # padded_img = np.pad(img, ((pr, pr), (pc, pc))) # wait this is wrong for fft
    out_r = r + kr - 1
    out_c = c + kc - 1
    # print("fft sizes:", out_r, out_c)
    
    img_f = np.fft.fft2(img, s=(out_r, out_c))
    K_f = np.fft.fft2(K, s=(out_r, out_c))
    
    out_f = img_f * K_f
    out = np.real(np.fft.ifft2(out_f))
    
    # chop off the edges
    return out[pr : pr + r, pc : pc + c]


def conv2d_separable(img, K, tol=1e-10):
    """TODO 1.3: rank-1 only. Raise if K is not rank-1."""
    U, S, Vh = np.linalg.svd(K)
    rank = np.sum(S > tol)
    if rank != 1:
        raise ValueError(f"Kernel is not rank-1 (rank={rank})")
        
    u = U[:, 0:1] * S[0]
    v = Vh[0:1, :]
    
    return conv2d_taps(conv2d_taps(img, v), u)


def conv2d_lowrank(img, K, r):
    """TODO 1.3: sum of r separable passes from the truncated SVD."""
    U, S, Vh = np.linalg.svd(K)
    
    out = np.zeros_like(img)
    for i in range(r):
        u = U[:, i:i+1] * S[i]
        v = Vh[i:i+1, :]
        out += conv2d_taps(conv2d_taps(img, v), u)
        
    return out


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from scipy.signal import convolve2d          # checking only
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:128, :128]
    K = kernel_bank(7)["gaussian"]
    ref = convolve2d(img, K, mode="same", boundary="fill")
    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        try:
            print(f"{fn.__name__:16s} max|err| = {np.abs(fn(img, K) - ref).max():.3e}")
        except NotImplementedError:
            print(f"{fn.__name__:16s} not implemented")

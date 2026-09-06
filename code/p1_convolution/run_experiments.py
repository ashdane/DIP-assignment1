import numpy as np
import time
from pathlib import Path
from PIL import Image
from starter import (
    kernel_bank, numeric_rank, conv2d_loops, conv2d_taps,
    conv2d_im2col, conv2d_fft, conv2d_separable, conv2d_lowrank, psnr
)
import matplotlib.pyplot as plt

def test_asymmetric():
    print("--- 1.1 Asymmetric Test ---")
    img = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    K = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    out = conv2d_taps(img, K)
    print("Output of convolution with impulse:\n", out)
    print("Flipped kernel:\n", np.flip(K))
    print("Is it equal to flipped kernel?", np.allclose(out, np.flip(K)))
    
def test_1_2_rank():
    print("\n--- 1.2 Kernel Rank ---")
    kernels = kernel_bank(k=15)
    tol = 1e-10
    print(f"Using threshold tol={tol} for singular values")
    for name, K in kernels.items():
        S = np.linalg.svd(K, compute_uv=False)
        rank = np.sum(S > tol)
        print(f"{name:15s} rank={rank:2d}")

def test_1_3_psnr():
    print("\n--- 1.3 Separability and Low Rank PSNR ---")
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:512, :512]
    kernels = kernel_bank(k=21)
    
    test_kernels = ["disk", "log", "motion_45deg", "random"]
    for name in test_kernels:
        K = kernels[name]
        exact = conv2d_taps(img, K)
        max_r = min(K.shape)
        psnrs = []
        for r in range(1, max_r + 1):
            approx = conv2d_lowrank(img, K, r)
            p = psnr(exact, approx)
            psnrs.append(p)
        print(f"PSNRs for {name}: {psnrs}")

def test_1_4_runtime():
    print("\n--- 1.4 Runtime Analysis ---")
    root = Path(__file__).resolve().parents[2]
    img_full = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)
    
    print("Sweep against k at 512x512")
    img_512 = img_full[:512, :512]
    k_vals = [3, 7, 11, 15, 21, 31]
    for k in k_vals:
        K = kernel_bank(k)["gaussian"]
        t0 = time.time()
        conv2d_taps(img_512, K)
        t1 = time.time()
        print(f"k={k:2d}: {t1-t0:.4f}s")
        
    print("\nSweep against N at k=15")
    K = kernel_bank(15)["gaussian"]
    N_vals = [128, 256, 512, 1024, 2048]
    for N in N_vals:
        img_N = img_full[:N, :N]
        t0 = time.time()
        conv2d_taps(img_N, K)
        t1 = time.time()
        h, w = N, N
        kh, kw = 15, 15
        mem_bytes = (h * w) * (kh * kw) * 8
        print(f"N={N:4d}: taps={t1-t0:.4f}s, im2col_peak_mem={mem_bytes/1024/1024:.2f}MB")

if __name__ == "__main__":
    test_asymmetric()
    test_1_2_rank()
    test_1_3_psnr()
    test_1_4_runtime()

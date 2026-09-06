import numpy as np
import time
from pathlib import Path
from PIL import Image
from starter import (
    bit_planes, reconstruct, gray_encode, embed_lsb, extract_lsb,
    embed_robust, extract_robust, degrade_gaussian, degrade_jpeg, ber, psnr
)
import matplotlib.pyplot as plt

def run_2_1():
    print("--- 2.1 Decomposition ---")
    root = Path(__file__).resolve().parents[2]
    tex = np.asarray(Image.open(root / "images/p2/cover_textured.png").convert("L"))
    planes = bit_planes(tex)
    
    psnrs = []
    for n in range(1, 9):
        recon = reconstruct(planes, n)
        p = psnr(tex, recon)
        psnrs.append((n, p))
    print("PSNR vs n (keep n planes):", psnrs)
    
def run_2_2():
    print("\n--- 2.2 Hide and recover a payload ---")
    root = Path(__file__).resolve().parents[2]
    stego = np.asarray(Image.open(root / "images/p2/decode_me.png").convert("L"))
    cover = np.asarray(Image.open(root / "images/p2/cover_textured.png").convert("L"))
    
    stego_flat = stego.flatten()
    bits = extract_lsb(stego, len(stego_flat), plane=0)
    
    sync = bits[:8]
    print("Sync marker:", sync)
    h_bits = bits[8:16]
    w_bits = bits[16:24]
    
    height = int("".join(str(b) for b in h_bits), 2)
    width = int("".join(str(b) for b in w_bits), 2)
    print(f"Payload dimensions: {width} x {height}")
    
    payload_bits = bits[24:24+height*width]
    payload_img = payload_bits.reshape((height, width)) * 255
    Image.fromarray(payload_img.astype(np.uint8)).save("payload.png")
    
    stego_psnr = psnr(cover, stego)
    print(f"Stego PSNR (decode_me vs cover): {stego_psnr:.2f} dB")
    
    smooth = np.asarray(Image.open(root / "images/p2/cover_smooth.png").convert("L"))
    
    rng = np.random.default_rng(0)
    test_payload = rng.integers(0, 2, 256*256, dtype=np.uint8)
    
    stego_tex = embed_lsb(cover, test_payload, plane=0)
    stego_smooth = embed_lsb(smooth, test_payload, plane=0)
    
    print(f"Textured PSNR (256x256 payload): {psnr(cover, stego_tex):.2f} dB")
    print(f"Smooth PSNR (256x256 payload): {psnr(smooth, stego_smooth):.2f} dB")
    
def run_2_3():
    print("\n--- 2.3 Survive the channel ---")
    root = Path(__file__).resolve().parents[2]
    cover = np.asarray(Image.open(root / "images/p2/cover_textured.png").convert("L"))
    
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 128, dtype=np.uint8)
    
    stego = embed_robust(cover, bits)
    stego_psnr = psnr(cover, stego)
    print(f"Robust Stego PSNR: {stego_psnr:.2f} dB")
    
    print("Evaluating against Gaussian noise:")
    for sigma in [0, 1, 2, 5, 10, 20]:
        noisy = degrade_gaussian(stego, sigma, rng=rng)
        extracted = extract_robust(noisy, cover, 128)
        error = ber(bits, extracted)
        print(f"Sigma={sigma:2d}, BER={error:.4f}")
        
    print("Evaluating against JPEG Q=75:")
    jpeg_stego = degrade_jpeg(stego, 75)
    extracted_jpeg = extract_robust(jpeg_stego, cover, 128)
    error_jpeg = ber(bits, extracted_jpeg)
    print(f"JPEG Q=75, BER={error_jpeg:.4f}")

if __name__ == "__main__":
    run_2_1()
    run_2_2()
    run_2_3()

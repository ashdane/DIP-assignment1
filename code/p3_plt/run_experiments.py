import numpy as np
import time
from pathlib import Path
from PIL import Image
from starter import transfer_curve, fit_piecewise, apply_recovered, support_report
import matplotlib.pyplot as plt

def run_3_1_and_3_2():
    print("--- 3.1 & 3.2 Recover Transforms ---")
    root = Path(__file__).resolve().parents[2]
    
    pairs = [
        ("A", "field_in.png", "field_out_A.png"),
        ("B", "field_in.png", "field_out_B.png"),
        ("C", "nebula_in.png", "nebula_out_C.png")
    ]
    
    for name, in_f, out_f in pairs:
        img_in = np.asarray(Image.open(root / f"images/p3/{in_f}").convert("L"))
        img_out = np.asarray(Image.open(root / f"images/p3/{out_f}").convert("L"))
        
        T, cnt = transfer_curve(img_in, img_out)
        
        # known for A and B (max_seg can be 3 for A, 4 for B, but we can just use 6 and let the penalty work)
        bps, slopes, inter, k = fit_piecewise(T, cnt, max_seg=6)
        
        print(f"\nTransform {name}:")
        print(f"Number of segments: {k}")
        print(f"Breakpoints: {bps}")
        print(f"Slopes: {slopes}")
        print(f"Intercepts: {inter}")
        
        recon_out = apply_recovered(img_in, bps, slopes, inter)
        
        max_err = np.max(np.abs(img_out.astype(int) - recon_out.astype(int)))
        mean_err = np.mean(np.abs(img_out.astype(int) - recon_out.astype(int)))
        print(f"Max Absolute Error: {max_err}")
        print(f"Mean Absolute Error: {mean_err:.4f}")
        
        if name == "B":
            print("Why B is recoverable from pair but not invertible from output alone:")
            print("B is non-monotonic (grey-level slicing). Multiple input intensities map to the same output intensity. Without the input image, we can't tell which input generated a specific output.")

def run_3_3():
    print("\n--- 3.3 Identifiability ---")
    root = Path(__file__).resolve().parents[2]
    img_in = np.asarray(Image.open(root / "images/p3/field_skycrop_in.png").convert("L"))
    img_out = np.asarray(Image.open(root / "images/p3/field_skycrop_out_D.png").convert("L"))
    
    T, cnt = transfer_curve(img_in, img_out)
    
    unsupported = support_report(cnt, threshold=10)
    print(f"Unsupported intensity ranges (<10 pixels): {unsupported}")
    
    bps, slopes, inter, k = fit_piecewise(T, cnt, max_seg=4)
    print(f"Segments fitted: {k}")
    print(f"Breakpoints: {bps}")
    print(f"Slopes: {slopes}")
    
    print("Why fewer than 4 segments were reported:")
    print("Because the data is only supported in a narrow band (the sky). There is simply no information outside this band to fit other segments. Any transform that agrees within the supported range fits equally well.")

if __name__ == "__main__":
    run_3_1_and_3_2()
    run_3_3()

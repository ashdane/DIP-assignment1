import numpy as np
import time
from pathlib import Path
from PIL import Image
import cv2
from starter import (
    to_ycbcr, key_naive_rgb, estimate_key_colour, key_soft,
    suppress_spill, composite, ChromaLUT,
    sad, mse_alpha, grad_error, temporal_flicker
)

def run_p5():
    root = Path(__file__).resolve().parents[2]
    p5_dir = root / "images" / "p5"
    
    # 1. Read video frames
    vid_path = str(p5_dir / "greenscreen.mp4")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise Exception("Failed to open greenscreen.mp4")
        
    frames = []
    for _ in range(48):
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR, convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if len(frames) == 0:
        raise Exception("No frames read from video.")
        
    H, W, _ = frames[0].shape
    print(f"Loaded {len(frames)} frames of size {W}x{H}")
    
    # 2. Read background plate
    bg_path = str(p5_dir / "background.jpg")
    bg_img = np.asarray(Image.open(bg_path).convert("RGB"))
    bg_H, bg_W, _ = bg_img.shape
    print(f"Loaded background of size {bg_W}x{bg_H}")
    
    if bg_H < H:
        print("Warning: Background height is smaller than frame height.")
        
    # 3. Naive key on first frame
    frame0 = frames[0]
    alpha_naive = key_naive_rgb(frame0)
    print("\n--- 5.1 Naive key ---")
    print("Generated naive mask.")
    # Failure analysis: find a pixel where naive fails (e.g. green backing is darker, G is not > R+60)
    
    # 4. Soft matte
    print("\n--- 5.2 Soft matte and spill suppression ---")
    key_color = estimate_key_colour(frame0)
    print(f"Estimated key colour (Cb, Cr): {key_color}")
    
    # Let's compute distance distributions for frame0
    ycbcr0 = to_ycbcr(frame0)
    cbcr0 = ycbcr0[:, :, 1:]
    dist = np.linalg.norm(cbcr0 - key_color, axis=2)
    
    print("Distance distributions for frame 0:")
    print(f"Min distance: {dist.min():.2f}")
    print(f"Max distance: {dist.max():.2f}")
    print(f"Median distance: {np.median(dist):.2f}")
    
    # Choose t_in and t_out based on typical distances
    t_in = 10.0
    t_out = 40.0
    
    alpha_soft = key_soft(frame0, key_cbcr=key_color, t_in=t_in, t_out=t_out)
    frac_pixels = np.sum((alpha_soft > 0.0) & (alpha_soft < 1.0)) / (H * W)
    print(f"Fraction of pixels with fractional alpha: {frac_pixels:.4f}")
    
    # Matte comparisons
    alpha_diff = np.mean(np.abs(alpha_naive - alpha_soft))
    g_err = grad_error(alpha_naive, alpha_soft)
    print("Matte difference between Naive and Soft:")
    print(f"Mean absolute difference: {alpha_diff:.4f}")
    print(f"Gradient field difference: {g_err:.4f}")
    
    # 5. Video compositing with ChromaLUT
    print("\n--- 5.3 Video, efficiently ---")
    lut = ChromaLUT(key_cbcr=key_color, t_in=t_in, t_out=t_out)
    
    # Prepare panning backgrounds
    # If bg_W >= 2.5*W, we can pan W*1.5 over 48 frames -> step = (bg_W - W) / 48
    pan_step = max(0, (bg_W - W) / len(frames))
    
    out_frames = []
    alphas = []
    
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        # Profiling to find bottleneck
        t0 = time.time()
        
        # 1. Alpha
        a = lut.alpha(frame)
        t_alpha = time.time()
        
        # 2. Suppress spill
        fg_despill = suppress_spill(frame, a, strength=1.0)
        t_spill = time.time()
        
        # 3. BG pan
        x_offset = int(i * pan_step)
        # Ensure we don't go out of bounds
        if x_offset + W > bg_W: x_offset = bg_W - W
        bg_crop = bg_img[:H, x_offset:x_offset+W]
        
        # Handle cases where BG is smaller in height
        if bg_crop.shape[0] < H:
            pad_h = H - bg_crop.shape[0]
            bg_crop = np.pad(bg_crop, ((0, pad_h), (0, 0), (0, 0)), mode='edge')
            
        t_bg = time.time()
        
        # 4. Composite
        comp = composite(fg_despill, bg_crop, a, spill=False) # spill already done
        t_comp = time.time()
        
        out_frames.append(comp)
        alphas.append(a)
        
        if i == 0:
            print("Bottleneck profile for frame 0 (seconds):")
            print(f"  Alpha LUT:  {t_alpha - t0:.4f}")
            print(f"  Despill:    {t_spill - t_alpha:.4f}")
            print(f"  BG crop:    {t_bg - t_spill:.4f}")
            print(f"  Composite:  {t_comp - t_bg:.4f}")
            
    total_time = time.time() - start_time
    fps = len(frames) / total_time
    print(f"\nProcessed {len(frames)} frames in {total_time:.2f}s ({fps:.2f} FPS)")
    
    # Temporal flicker
    flicker_raw = temporal_flicker(alphas)
    
    # Smooth temporally
    # Simple Exponential Moving Average (EMA)
    smooth_alphas = [alphas[0]]
    for i in range(1, len(alphas)):
        smooth = 0.6 * alphas[i] + 0.4 * smooth_alphas[-1]
        smooth_alphas.append(smooth)
        
    flicker_smooth = temporal_flicker(smooth_alphas)
    print(f"\nTemporal flicker (raw): {flicker_raw:.4f}")
    print(f"Temporal flicker (smoothed): {flicker_smooth:.4f}")
    
    # Save video
    out_path = str(p5_dir / "composite_out.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (W, H))
    for f in out_frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved video to {out_path}")
    
    # Save a few frames for report
    Image.fromarray((alpha_naive * 255).astype(np.uint8)).save(p5_dir / "naive_alpha.png")
    Image.fromarray((alpha_soft * 255).astype(np.uint8)).save(p5_dir / "soft_alpha.png")
    Image.fromarray(out_frames[0]).save(p5_dir / "composite_frame0.png")

if __name__ == "__main__":
    run_p5()

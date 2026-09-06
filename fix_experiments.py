import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent

# --- 4.3(e) Local histogram specification ---
def p4_3e():
    import sys
    sys.path.append(str(root / "code" / "p4_histogram"))
    from starter import specify, clahe
    
    img = np.asarray(Image.open(root / "images" / "p4" / "local_regions.png").convert("RGB"))
    # BT.601 luma
    Y = np.round(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
    
    # Extract a dark tile
    h, w = Y.shape
    tile = Y[int(h*0.7):int(h*0.9), int(w*0.1):int(w*0.3)]
    
    # Target: A Gaussian with higher mean to brighten it without making it look completely washed out
    target = np.exp(-0.5 * ((np.arange(256) - 160)/30)**2)
    spec_tile = specify(tile, target)
    
    # Compare with CLAHE
    clahe_full = clahe(Y, tiles=8, clip=3.0)
    clahe_tile = clahe_full[int(h*0.7):int(h*0.9), int(w*0.1):int(w*0.3)]
    
    # Save the patch comparisons
    out_dir = root / "images" / "p4" / "p4_3e_local_spec"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(tile).save(out_dir / "original_dark_tile.png")
    Image.fromarray(spec_tile).save(out_dir / "specified_tile.png")
    Image.fromarray(clahe_tile).save(out_dir / "clahe_tile.png")
    print("4.3(e) done.")

# --- 5.1 Naive key failing pixel ---
# --- 5.2(a) Chroma distance distributions ---
def p5_fixes():
    import sys
    if 'starter' in sys.modules: del sys.modules['starter']
    sys.path.remove(str(root / "code" / "p4_histogram"))
    sys.path.append(str(root / "code" / "p5_greenscreen"))
    from starter import estimate_key_colour, to_ycbcr, key_naive_rgb
    
    cap = cv2.VideoCapture(str(root / "images" / "p5" / "greenscreen.mp4"))
    ret, frame = cap.read()
    cap.release()
    if not ret: return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 5.1 Naive key failing pixel
    alpha_naive = key_naive_rgb(frame)
    # Find a pixel where alpha_naive == 1 (fg) but we know it's bg.
    # Typically in corners or shadows.
    # Let's search the top-left 50x50 region which should be background
    bg_region = alpha_naive[:50, :50]
    fails = np.where(bg_region == 1.0)
    if len(fails[0]) > 0:
        fy, fx = fails[0][0], fails[1][0]
        r, g, b = frame[fy, fx]
        print(f"5.1 Failing pixel at (y={fy}, x={fx}): R={r}, G={g}, B={b}")
        print(f"Naive key condition: G > R+60 ({g} > {r+60}) and G > B+60 ({g} > {b+60})")
    
    # 5.2(a) Chroma distance distributions
    key_color = estimate_key_colour(frame)
    ycbcr = to_ycbcr(frame)
    cbcr = ycbcr[:, :, 1:]
    dist = np.linalg.norm(cbcr - key_color, axis=2).ravel()
    
    # To separate populations, we need to segment.
    # We can use our soft key thresholds:
    # < 10: pure backing
    # 10 to 40: partially transparent
    # > 40: pure foreground
    
    pure_bg = dist[dist < 10]
    transparent = dist[(dist >= 10) & (dist <= 40)]
    pure_fg = dist[dist > 40]
    
    plt.figure()
    plt.hist(pure_bg, bins=20, alpha=0.5, label='Pure Backing', color='green')
    plt.hist(transparent, bins=20, alpha=0.5, label='Partially Transparent', color='yellow')
    plt.hist(pure_fg, bins=50, alpha=0.5, label='Pure Foreground', color='red')
    plt.xlabel('Chroma Distance from Key Colour')
    plt.ylabel('Pixel Count')
    plt.title('Chroma Distance Distributions')
    plt.legend()
    plt.yscale('log') # Log scale because BG dominates
    plt.savefig(root / "plots" / "p5_chroma_dists.png")
    plt.close()
    print("5.2(a) done.")

if __name__ == "__main__":
    p4_3e()
    p5_fixes()

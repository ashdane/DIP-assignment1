import numpy as np
from pathlib import Path
from PIL import Image
from starter import hist, cdf, equalise, specify, wasserstein1, ahe, clahe, apply_on_luma, auto_correct

def rgb2luma(rgb):
    rgb_f = rgb.astype(np.float64)
    Y = np.round(0.299 * rgb_f[:,:,0] + 0.587 * rgb_f[:,:,1] + 0.114 * rgb_f[:,:,2]).astype(np.uint8)
    return Y

def run_4_1():
    print("--- 4.1 Global equalisation ---")
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p4/equalization/input.png").convert("L"))
    
    eq = equalise(img)
    # Why equalised histogram is not flat:
    print("4.1(a) Proof why equalised histogram is not flat:")
    print("Because equalisation on discrete integer levels acts as a point operation (a LUT). It can only merge bins or stretch empty spaces between bins, but it cannot split a single bin's mass across multiple output bins. The histogram peaks remain intact, just shifted.")
    
    # Idempotence
    eq2 = equalise(eq)
    print("\n4.1(b) Idempotence:")
    print("Equalisation is idempotent because once the CDF is roughly linear (i.e., F(v) = v/255), applying equalisation again maps v to round(255 * v/255) = v. The mapping becomes the identity function.")
    print("Max diff between equalise(img) and equalise(equalise(img)):", np.max(np.abs(eq.astype(int) - eq2.astype(int))))

    print("\n4.1(c) Image where equalisation does nothing:")
    print("An image that already has a perfectly flat histogram (uniform distribution of intensities) will have a linear CDF F(v) = v/255. Applying equalisation gives T(v) = round(255 * v/255) = v, which does nothing.")

def run_4_2():
    print("\n--- 4.2 Histogram matching and specification ---")
    root = Path(__file__).resolve().parents[2]
    
    print("\n(a) Image matching:")
    src = np.asarray(Image.open(root / "images/p4/matching/source.png").convert("L"))
    ref = np.asarray(Image.open(root / "images/p4/matching/reference.png").convert("L"))
    h_ref = hist(ref)
    
    w1_before = wasserstein1(hist(src), h_ref)
    matched = specify(src, h_ref)
    w1_after = wasserstein1(hist(matched), h_ref)
    print(f"W1 before matching: {w1_before:.4f}")
    print(f"W1 after matching: {w1_after:.4f}")
    
    print("\n(b) Histogram specification:")
    src_spec = np.asarray(Image.open(root / "images/p4/specification/source.png").convert("L"))
    uni_target = np.full(256, 1.0)
    w1_uni_before = wasserstein1(hist(src_spec), uni_target)
    uni_matched = specify(src_spec, uni_target)
    w1_uni_after = wasserstein1(hist(uni_matched), uni_target)
    print(f"Uniform target W1 before: {w1_uni_before:.4f}, after: {w1_uni_after:.4f}")
    print("Difference between uniform specification and plain equalisation:")
    print("Essentially none. Both target a uniform distribution, producing the same CDF (a straight line from 0 to 1).")
    
    v = np.arange(256)
    mu, sigma = 128, 35
    gauss_target = np.exp(-0.5 * ((v - mu)/sigma)**2)
    w1_gauss_before = wasserstein1(hist(src_spec), gauss_target)
    gauss_matched = specify(src_spec, gauss_target)
    w1_gauss_after = wasserstein1(hist(gauss_matched), gauss_target)
    print(f"Gaussian target W1 before: {w1_gauss_before:.4f}, after: {w1_gauss_after:.4f}")
    
    print("\n(c) Colour:")
    rgb_src = np.asarray(Image.open(root / "images/p4/colour_source.png").convert("RGB"))
    
    def rgb_hue(rgb):
        # simple hue for scoring: np.arctan2(sqrt(3)*(G-B), 2R-G-B)
        rgb = rgb.astype(float)
        R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
        num = np.sqrt(3) * (G - B)
        den = 2*R - G - B
        hue = np.arctan2(num, den)
        sat = np.sqrt(num**2 + den**2) / (np.sqrt(6)*255 + 1e-6)
        return hue, sat
        
    def hue_dist(h1, h2):
        d = np.abs(h1 - h2)
        return np.minimum(d, 2*np.pi - d)
    
    rgb_eq = np.zeros_like(rgb_src)
    for c in range(3):
        rgb_eq[..., c] = equalise(rgb_src[..., c])
        
    luma_eq = apply_on_luma(rgb_src, equalise)
    
    h1, s1 = rgb_hue(rgb_src)
    h2, _ = rgb_hue(rgb_eq)
    h3, _ = rgb_hue(luma_eq)
    
    mask = s1 > 0.05 # saturation threshold
    
    shift_rgb = np.mean(hue_dist(h1[mask], h2[mask]))
    shift_luma = np.mean(hue_dist(h1[mask], h3[mask]))
    
    print(f"Mean Hue shift (per-channel): {shift_rgb:.4f} radians")
    print(f"Mean Hue shift (luma-only): {shift_luma:.4f} radians")
    print("Why per-channel is wrong:")
    print("Equalisation assumes single-channel luminance data. When applied independently to R, G, and B, it alters their ratios, which directly changes the hue and ruins the colour balance.")

def run_4_3():
    print("\n--- 4.3 Local histogram processing ---")
    print("(b) AHE failures:")
    print("1. Noise amplification: In flat/homogeneous regions, AHE stretches tiny noise fluctuations across the entire intensity range, making the tiles extremely noisy.")
    print("2. Tiling artifacts: Because there is no interpolation, the boundaries between adjacent tiles with different histograms show stark, visible grid lines.")
    
    print("\n(d) CLAHE ablation limits:")
    print("Clip limit = 1: Degenerates to no equalisation (identity transform) because no bin can exceed the mean, forcing the PDF to be uniform before CDF calculation.")
    print("Clip limit = inf: Degenerates to plain AHE, amplifying noise massively in flat regions.")
    print("Tile count = 2 (very low): Acts almost like global equalisation, missing local detail.")
    print("Tile count = 64 (very high): Over-enhances extremely small regions, destroying global contrast and generating excessive local noise.")

def run_4_4():
    print("\n--- 4.4 Automatic correction ---")
    root = Path(__file__).resolve().parents[2]
    
    evs = []
    for i in range(5):
        img = np.asarray(Image.open(root / f"images/p4/exposure/ev{i}.png").convert("RGB"))
        evs.append(img)
        
    lumas = [rgb2luma(img) for img in evs]
    
    def calc_mean_max_w1(l_list):
        dists = []
        for i in range(5):
            for j in range(i+1, 5):
                d = wasserstein1(hist(l_list[i]), hist(l_list[j]))
                dists.append(d)
        return np.mean(dists), np.max(dists)
        
    mean_b, max_b = calc_mean_max_w1(lumas)
    print(f"W1 before correction - Mean: {mean_b:.4f}, Max: {max_b:.4f}")
    
    corrected = [auto_correct(img) for img in evs]
    corr_lumas = [rgb2luma(img) for img in corrected]
    
    mean_a, max_a = calc_mean_max_w1(corr_lumas)
    print(f"W1 after correction - Mean: {mean_a:.4f}, Max: {max_a:.4f}")
    
    print("\nOutput stats (StdDev, 1st-99th percentile):")
    for i, luma in enumerate(corr_lumas):
        std = np.std(luma)
        p1 = np.percentile(luma, 1)
        p99 = np.percentile(luma, 99)
        print(f"EV{i}: Std={std:.2f}, Range=[{p1:.0f}, {p99:.0f}]")
        
    print("\nEstimator and failure case:")
    print("Estimator: 1st and 99th percentiles are mapped to 0 and 255. Failure case: Images containing a large bright light source (e.g., the sun) covering >1% of pixels, or vast black space (like astronomy photos) covering >1% of pixels, will skew the percentiles and wash out the midtones.")

if __name__ == "__main__":
    run_4_1()
    run_4_2()
    run_4_3()
    run_4_4()

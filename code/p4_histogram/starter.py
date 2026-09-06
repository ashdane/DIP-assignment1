"""P4 starter. Implement every function marked TODO.

Dataset map (relative to the bundle root):
  4.1  images/p4/equalization/input.png
  4.2  images/p4/matching/{source,reference}.png
       images/p4/specification/source.png
       images/p4/colour_source.png
  4.3  images/p4/local_regions.png
  4.4  images/p4/exposure/ev0.png ... ev4.png
"""
import numpy as np


def hist(img, bins=256):
    return np.bincount(img.ravel(), minlength=bins).astype(np.float64)


def cdf(h):
    """TODO 4.1: normalised cumulative histogram."""
    c = np.cumsum(h)
    return c / c[-1]


def equalise(img):
    """TODO 4.1: CDF-based global equalisation."""
    h = hist(img)
    F = cdf(h)
    LUT = np.round(255 * F).astype(np.uint8)
    return LUT[img]


def specify(img, target_hist):
    """TODO 4.2: match img's histogram to target_hist."""
    h_in = hist(img)
    F_in = cdf(h_in)
    F_tgt = cdf(target_hist)
    
    LUT = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        idx = np.searchsorted(F_tgt, F_in[v])
        if idx == 256:
            idx = 255
        LUT[v] = idx
    return LUT[img]


def wasserstein1(h1, h2):
    """TODO 4.2: W1 distance between two histograms. Your scoring metric."""
    F1 = cdf(h1)
    F2 = cdf(h2)
    return np.sum(np.abs(F1 - F2))


def ahe(img, tiles=8):
    """TODO 4.3: plain tiled AHE. No clipping, no interpolation. This one is
    SUPPOSED to look bad -- that is the point."""
    rows, cols = img.shape
    out = np.zeros_like(img)
    th = rows // tiles
    tw = cols // tiles
    
    for i in range(tiles):
        for j in range(tiles):
            r_start = i * th
            r_end = (i + 1) * th if i < tiles - 1 else rows
            c_start = j * tw
            c_end = (j + 1) * tw if j < tiles - 1 else cols
            
            patch = img[r_start:r_end, c_start:c_end]
            out[r_start:r_end, c_start:c_end] = equalise(patch)
            
    return out


def clahe(img, tiles=8, clip=3.0, bins=256):
    """TODO 4.3: clip at `clip` x mean bin height, redistribute the excess,
    and bilinearly interpolate between the four surrounding tile LUTs.

    Watch the tile-centre offset. That is where the marks go."""
    r, c = img.shape
    th = r / tiles
    tw = c / tiles
    
    luts = np.zeros((tiles, tiles, 256), dtype=np.float64)
    for i in range(tiles):
        for j in range(tiles):
            r_start = int(i * th)
            r_end = int((i + 1) * th if i < tiles - 1 else r)
            c_start = int(j * tw)
            c_end = int((j + 1) * tw if j < tiles - 1 else c)
            patch = img[r_start:r_end, c_start:c_end]
            
            hh = np.bincount(patch.ravel(), minlength=256).astype(np.float64)
            mean_height = len(patch.ravel()) / 256.0
            clip_limit = clip * mean_height
            
            excess = 0.0
            for b in range(256):
                if hh[b] > clip_limit:
                    excess += hh[b] - clip_limit
                    hh[b] = clip_limit
            
            hh += excess / 256.0
            F = np.cumsum(hh) / np.sum(hh)
            luts[i, j] = np.round(255 * F)
            
    yy, xx = np.meshgrid(np.arange(r), np.arange(c), indexing='ij')
    
    # this took forever to figure out
    # ty = yy / th - wait no
    # why is the offset th/2??
    ty = (yy - th / 2) / th
    tx = (xx - tw / 2) / tw
    
    ty = np.clip(ty, 0, tiles - 1.001)
    tx = np.clip(tx, 0, tiles - 1.001)
    
    top_y = np.floor(ty).astype(int)
    bot_y = np.clip(top_y + 1, 0, tiles - 1)
    left_x = np.floor(tx).astype(int)
    right_x = np.clip(left_x + 1, 0, tiles - 1)
    
    y_dist = ty - top_y
    x_dist = tx - left_x
    
    v = img
    val00 = luts[top_y, left_x, v]
    val01 = luts[top_y, right_x, v]
    val10 = luts[bot_y, left_x, v]
    val11 = luts[bot_y, right_x, v]
    
    val0 = val00 * (1 - x_dist) + val01 * x_dist
    val1 = val10 * (1 - x_dist) + val11 * x_dist
    val = val0 * (1 - y_dist) + val1 * y_dist
    
    return np.round(val).astype(np.uint8)


def apply_on_luma(rgb, fn):
    """TODO 4.2: run a greyscale operator on luma only, preserving hue."""
    rgb_f = rgb.astype(np.float64)
    Y = np.round(0.299 * rgb_f[:,:,0] + 0.587 * rgb_f[:,:,1] + 0.114 * rgb_f[:,:,2]).astype(np.uint8)
    
    Y_f = np.maximum(Y.astype(np.float64), 1e-6)
    Y_new = fn(Y)
    
    scale = Y_new.astype(np.float64) / Y_f
    out = rgb_f * scale[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def auto_correct(rgb):
    """TODO 4.4: no parameters. Estimate what you need from the image."""
    def stretch(Y):
        p1 = np.percentile(Y, 1)
        p99 = np.percentile(Y, 99)
        if p99 == p1: return Y
        Y_new = (Y.astype(float) - p1) / (p99 - p1) * 255
        return np.clip(np.round(Y_new), 0, 255).astype(np.uint8)
    return apply_on_luma(rgb, stretch)

"""P2 starter. Implement every function marked TODO."""
import numpy as np


def bit_planes(img):
    """TODO 2.1: (H,W) uint8 -> (8,H,W) uint8 in {0,1}, index 0 = LSB."""
    img = np.asarray(img, dtype=np.uint8)
    return np.array([(img >> i) & 1 for i in range(8)], dtype=np.uint8)


def reconstruct(planes, keep):
    """TODO 2.1: rebuild from the `keep` most significant planes."""
    out = np.zeros(planes.shape[1:], dtype=np.uint8)
    for i in range(8 - keep, 8):
        out |= (planes[i] << i)
    return out


def gray_encode(img):
    """TODO 2.1: Gray-coded intensities, for the ramp comparison."""
    img = np.asarray(img, dtype=np.uint8)
    return img ^ (img >> 1)


def embed_lsb(cover, bits, plane=0):
    """TODO 2.2: write bits into the given plane, raster order."""
    stego = cover.copy()
    h, w = stego.shape
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    n = len(bits)
    stego_flat = stego.flatten()
    mask = np.uint8(255 - (1 << plane))
    stego_flat[:n] = (stego_flat[:n] & mask) | (bits << plane)
    return stego_flat.reshape(h, w)


def extract_lsb(stego, n, plane=0):
    """TODO 2.2: read n bits back out."""
    stego_flat = stego.flatten()
    bits = (stego_flat[:n] >> plane) & 1
    return bits.astype(np.uint8)

def embed_robust(cover, bits, block_size=64, delta=3, **kw):
    """TODO 2.3: your design. Must carry 128 bits at PSNR(cover,stego) >= 40 dB
    and survive the channels below. Document your parameters."""
    stego = np.array(cover, dtype=np.float64)
    h, w = cover.shape
    bits = np.asarray(bits).flatten()
    n = len(bits)
    blocks_per_row = w // block_size
    for i in range(n):
        row = (i // blocks_per_row) * block_size
        col = (i % blocks_per_row) * block_size
        sign = 1 if bits[i] == 1 else -1
        stego[row:row+block_size, col:col+block_size] += sign * delta
    return np.clip(np.round(stego), 0, 255).astype(np.uint8)

def extract_robust(stego, cover, n, block_size=64, **kw):
    """TODO 2.3: matching extractor."""
    bits = np.zeros(n, dtype=np.uint8)
    h, w = cover.shape
    blocks_per_row = w // block_size
    stego_f = stego.astype(np.float64)
    cover_f = cover.astype(np.float64)
    for i in range(n):
        row = (i // blocks_per_row) * block_size
        col = (i % blocks_per_row) * block_size
        diff = np.mean(stego_f[row:row+block_size, col:col+block_size] - 
                       cover_f[row:row+block_size, col:col+block_size])
        bits[i] = 1 if diff > 0 else 0
    return bits


# ---- provided channels: do not modify, these are what you are graded against
def degrade_gaussian(img, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(np.round(img.astype(np.float64) + rng.normal(0, sigma, img.shape)),
                   0, 255).astype(np.uint8)


def degrade_jpeg(img, quality):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def ber(a, b):
    return float(np.mean(a != b))


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)

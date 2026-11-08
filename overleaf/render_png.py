"""Render page 1 of a compiled pseudocode PDF to a PNG cropped to the ink, for the demos' panels.

Usage: uv run --with pymupdf python render_png.py <in.pdf> <out.png>
"""
import sys

import numpy as np
import matplotlib.image as mpimg
import pymupdf

DPI = 300
PAD = 30   # white border kept around the ink (pixels)

src, dst = sys.argv[1], sys.argv[2]
pix = pymupdf.open(src)[0].get_pixmap(dpi=DPI)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
ys, xs = np.nonzero((img < 128).any(axis=2))
crop = img[max(ys.min() - PAD, 0):ys.max() + PAD, max(xs.min() - PAD, 0):xs.max() + PAD]
mpimg.imsave(dst, crop)
print(f"{dst}: {crop.shape[1]}x{crop.shape[0]} px")

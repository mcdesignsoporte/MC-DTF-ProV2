import numpy as np
from PIL import Image, ImageOps


def make_halftone(img: Image.Image, dot_size: int = 8, angle: float = 15, invert: bool = False) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    base.alpha_composite(rgba)
    gray = ImageOps.grayscale(base)
    rotated = gray.rotate(angle, expand=True, fillcolor=255)
    pixels = np.array(rotated)
    h, w = pixels.shape
    out = np.ones((h, w), dtype=np.uint8) * 255
    step = max(4, int(dot_size))
    for y in range(0, h, step):
        for x in range(0, w, step):
            block = pixels[y:min(y+step, h), x:min(x+step, w)]
            darkness = 255 - float(block.mean())
            radius = (darkness / 255.0) * (step / 2.0)
            bh, bw = block.shape
            yy, xx = np.ogrid[:bh, :bw]
            mask = (yy - bh/2) ** 2 + (xx - bw/2) ** 2 <= radius ** 2
            out[y:y+bh, x:x+bw][mask] = 0
    out_img = Image.fromarray(out, "L").rotate(-angle, expand=True, fillcolor=255)
    left = max(0, (out_img.width - img.width) // 2)
    top = max(0, (out_img.height - img.height) // 2)
    out_img = out_img.crop((left, top, left + img.width, top + img.height))
    if invert:
        out_img = ImageOps.invert(out_img)
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    transparent = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask = ImageOps.invert(out_img) if not invert else out_img
    result = Image.composite(black, transparent, mask)
    result.putalpha(Image.fromarray(np.where(alpha > 0, 255, 0).astype(np.uint8)))
    return result

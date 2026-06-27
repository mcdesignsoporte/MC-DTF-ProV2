import numpy as np
from PIL import Image, ImageOps


def make_halftone(img: Image.Image, dot_size: int = 8, angle: int = 15, invert: bool = False) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    base.alpha_composite(rgba)
    gray = ImageOps.grayscale(base)
    rotated = gray.rotate(angle, expand=True, fillcolor=255)
    pix = np.array(rotated)
    h, w = pix.shape
    step = max(4, int(dot_size))
    out = np.ones((h, w), dtype=np.uint8) * 255

    for y in range(0, h, step):
        for x in range(0, w, step):
            block = pix[y:min(y+step, h), x:min(x+step, w)]
            darkness = 255 - float(block.mean())
            radius = (darkness / 255.0) * (step / 2)
            bh, bw = block.shape
            yy, xx = np.ogrid[:bh, :bw]
            mask = (yy - bh / 2) ** 2 + (xx - bw / 2) ** 2 <= radius ** 2
            out[y:y+bh, x:x+bw][mask] = 0

    mask_img = Image.fromarray(out, "L").rotate(-angle, expand=True, fillcolor=255)
    left = (mask_img.width - rgba.width) // 2
    top = (mask_img.height - rgba.height) // 2
    mask_img = mask_img.crop((left, top, left + rgba.width, top + rgba.height))
    if invert:
        mask_img = ImageOps.invert(mask_img)
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    transparent = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    result = Image.composite(black, transparent, ImageOps.invert(mask_img))

    dot_alpha = np.array(result.getchannel("A"), dtype=np.uint16)
    source_alpha = alpha.astype(np.uint16)
    combined_alpha = ((dot_alpha * source_alpha) // 255).astype(np.uint8)
    result.putalpha(Image.fromarray(combined_alpha, "L"))
    return result

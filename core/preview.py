from PIL import Image


def composite_preview(img: Image.Image, mode: str = "Gris") -> Image.Image:
    rgba = img.convert("RGBA")
    mode = (mode or "Gris").lower()
    if "negro" in mode:
        bg_color = (0, 0, 0, 255)
    elif "blanco" in mode:
        bg_color = (255, 255, 255, 255)
    elif "transparente" in mode:
        # Checkerboard
        tile = 32
        bg = Image.new("RGBA", rgba.size, (220, 220, 220, 255))
        pix = bg.load()
        for y in range(rgba.height):
            for x in range(rgba.width):
                if ((x // tile) + (y // tile)) % 2 == 0:
                    pix[x, y] = (170, 170, 170, 255)
        bg.alpha_composite(rgba)
        return bg
    else:
        bg_color = (128, 128, 128, 255)
    bg = Image.new("RGBA", rgba.size, bg_color)
    bg.alpha_composite(rgba)
    return bg

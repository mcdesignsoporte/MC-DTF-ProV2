from PIL import Image


def composite_preview(img: Image.Image, mode: str = "Transparente") -> Image.Image:
    img = img.convert("RGBA")
    m = (mode or "").lower()
    if "negro" in m:
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    elif "blanco" in m:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    elif "gris" in m:
        bg = Image.new("RGBA", img.size, (150, 150, 150, 255))
    else:
        # checkerboard
        tile = 24
        bg = Image.new("RGBA", img.size, (230, 230, 230, 255))
        px = bg.load()
        for y in range(img.height):
            for x in range(img.width):
                if ((x // tile) + (y // tile)) % 2:
                    px[x, y] = (190, 190, 190, 255)
    bg.alpha_composite(img)
    return bg.convert("RGB")

from PIL import Image


def checkerboard(size, square=20):
    w, h = size
    img = Image.new("RGB", size, (230, 230, 230))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            if ((x // square) + (y // square)) % 2 == 0:
                pix[x, y] = (190, 190, 190)
    return img


def composite_preview(img: Image.Image, mode: str = "Transparente") -> Image.Image:
    rgba = img.convert("RGBA")
    if mode == "Negro":
        bg = Image.new("RGB", rgba.size, (10, 10, 10))
    elif mode == "Blanco":
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
    elif mode == "Gris":
        bg = Image.new("RGB", rgba.size, (120, 120, 120))
    else:
        bg = checkerboard(rgba.size)
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg

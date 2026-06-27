from PIL import Image
import numpy as np

def has_transparency(img: Image.Image)->bool:
    if img.mode!="RGBA": return False
    alpha=np.array(img.getchannel("A"))
    return np.any(alpha<250)

def black_percentage(img):
    rgb=np.array(img.convert("RGB"))
    black=((rgb[:,:,0]<20)&(rgb[:,:,1]<20)&(rgb[:,:,2]<20))
    return float(black.mean())

def white_percentage(img):
    rgb=np.array(img.convert("RGB"))
    white=((rgb[:,:,0]>240)&(rgb[:,:,1]>240)&(rgb[:,:,2]>240))
    return float(white.mean())

def image_type(img):
    if has_transparency(img): return "png"
    if black_percentage(img)>0.55: return "dark_design"
    if white_percentage(img)>0.65: return "light_background"
    return "photo"

def detect(img):
    modes={
        "photo":{"name":"Fotografía","recommended":"photo","use_ai":True},
        "png":{"name":"PNG Transparente","recommended":"png","use_ai":False},
        "dark_design":{"name":"Diseño Oscuro","recommended":"dark","use_ai":False},
        "light_background":{"name":"Fondo Blanco","recommended":"white","use_ai":False},
    }
    return modes[image_type(img)]

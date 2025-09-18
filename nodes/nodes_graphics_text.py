#---------------------------------------------------------------------------------------------------------------------#
# Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# for ComfyUI                                                 https://github.com/comfyanonymous/ComfyUI
#---------------------------------------------------------------------------------------------------------------------#

import json
import numpy as np
import torch
import os
import platform
from PIL import Image, ImageDraw, ImageOps, ImageFont
from ..categories import icons
from ..config import color_mapping, COLORS
from .functions_graphics import *

'''
try:
    from bidi.algorithm import get_display
except ImportError:
    import subprocess
    subprocess.check_call(['python', '-m', 'pip', 'install', 'python_bidi'])

try:
    import arabic_reshaper
except ImportError:
    import subprocess
    subprocess.check_call(['python', '-m', 'pip', 'install', 'arabic_reshaper'])
'''

def get_offset_for_true_mm(text, draw, font):
    anchor_bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    anchor_center = (anchor_bbox[0] + anchor_bbox[2]) // 2, (anchor_bbox[1] + anchor_bbox[3]) // 2
    mask_bbox = font.getmask(text).getbbox()
    mask_center = (mask_bbox[0] + mask_bbox[2]) // 2, (mask_bbox[1] + mask_bbox[3]) // 2
    return anchor_center[0] - mask_center[0], anchor_center[1] - mask_center[1]


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

#---------------------------------------------------------------------------------------------------------------------#
def create_text_image(text, font_name, font_size, color, position_x, position_y, align, justify, margins, line_spacing, rotation_angle, rotation_options, canvas_size):
    """Returns a minimal RGBA PIL image of the text + its offset."""
    w, h = canvas_size
    text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    # Load font
    font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", font_name)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Failed to load font {font_name}: {e}")
        font = ImageFont.load_default()

    # Draw masked text (reusing your existing logic)
    mask = Image.new('L', (w, h), 0)
    rotated_mask = draw_masked_text(mask, text, font_name, font_size,
                                    margins, line_spacing,
                                    position_x, position_y,
                                    align, justify,
                                    rotation_angle, rotation_options)

    # Paste white text onto RGBA layer using the mask
    bbox = rotated_mask.getbbox()
    if not bbox:
        return None, (0, 0)

    # Extract minimal region
    cropped_mask = rotated_mask.crop(bbox)
    x0, y0, x1, y1 = bbox
    tw, th = x1 - x0, y1 - y0

    # Create minimal RGBA text image
    text_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(text_img)
    txt_draw.bitmap((0, 0), cropped_mask, fill=color)

    return text_img, (x0, y0)  # PIL image + top-left corner

#---------------------------------------------------------------------------------------------------------------------#

ALIGN_OPTIONS = ["center", "top", "bottom"]
ROTATE_OPTIONS = ["text center", "image center"]
JUSTIFY_OPTIONS = ["center", "left", "right"]
PERSPECTIVE_OPTIONS = ["top", "bottom", "left", "right"]

#---------------------------------------------------------------------------------------------------------------------#
class CR_OverlayText:

    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        return {"required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "text"}),
                "font_name": (file_list,),
                "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                "font_color": (COLORS,),
                "align": (ALIGN_OPTIONS,),
                "justify": (JUSTIFY_OPTIONS,),
                "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rotation_options": (ROTATE_OPTIONS,),
                },
                "optional": {"font_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE",)
    RETURN_NAMES = ("IMAGE", "show_help", "text_image",)
    FUNCTION = "overlay_text"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def overlay_text(self, image, text, font_name, font_size, font_color,
                 margins, line_spacing,
                 position_x, position_y,
                 align, justify,
                 rotation_angle, rotation_options,
                 font_color_hex='#000000'):

        # Get RGB values for the text color
        text_color = get_color_values(font_color, font_color_hex, color_mapping)

        # Convert tensor images
        image_3d = image[0, :, :, :]
        back_image = tensor2pil(image_3d)

        # Create PIL images for the text and background layers and text mask
        text_image = Image.new('RGB', back_image.size, text_color)
        text_mask = Image.new('L', back_image.size)

        # Draw the text on the text mask
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                            margins, line_spacing,
                                            position_x, position_y,
                                            align, justify,
                                            rotation_angle, rotation_options)

        # Composite the text image onto the background image using the rotated text mask
        image_out = Image.composite(text_image, back_image, rotated_text_mask)

        # --- EXTRACT MINIMAL RGBA TEXT IMAGE ---
        bbox = rotated_text_mask.getbbox()
        if bbox:
            tx, ty, tw, th = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            cropped_mask = rotated_text_mask.crop(bbox)
            rgba_text_pil = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            rgba_draw = ImageDraw.Draw(rgba_text_pil)
            rgba_draw.bitmap((0, 0), cropped_mask, fill=text_color)
        else:
            tx = ty = tw = th = 0
            rgba_text_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        text_tensor = pil2tensor(rgba_text_pil)  # [1, H, W, C]
        text_hash = get_tensor_hash(text_tensor)

        # --- BUILD SCENE GRAPH ---
        bg_hash = get_tensor_hash(image)

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": back_image.width,
            "height": back_image.height,
            "images": [
                # Background image
                {
                    "x": 0,
                    "y": 0,
                    "width": back_image.width,
                    "height": back_image.height,
                    "images": bg_hash
                },
                # Text overlay (real image tensor)
                {
                    "x": tx,
                    "y": ty,
                    "width": tw,
                    "height": th,
                    "images": text_hash
                }
            ]
        }

        show_help = json.dumps(show_help_data, indent=2)

        return (pil2tensor(image_out), show_help, text_tensor)
#---------------------------------------------------------------------------------------------------------------------#
class CR_DrawText:

    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        return {"required": {
                    "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                    "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                    "text": ("STRING", {"multiline": True, "default": "text"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "font_color": (COLORS,),
                    "background_color": (COLORS,),
                    "align": (ALIGN_OPTIONS,),
                    "justify": (JUSTIFY_OPTIONS,),
                    "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                    "rotation_options": (ROTATE_OPTIONS,),
                },
                "optional": {
                    "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                    "bg_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("IMAGE", "show_help",)
    FUNCTION = "draw_text"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def draw_text(self, image_width, image_height, text,
                  font_name, font_size, font_color,
                  background_color,
                  margins, line_spacing,
                  position_x, position_y,
                  align, justify,
                  rotation_angle, rotation_options,
                  font_color_hex='#000000', bg_color_hex='#000000'):

        # Get RGB values for the text and background colors
        text_color = get_color_values(font_color, font_color_hex, color_mapping)
        bg_color = get_color_values(background_color, bg_color_hex, color_mapping)

        # Create PIL images for the text and background layers and text mask
        size = (image_width, image_height)
        text_image = Image.new('RGB', size, text_color)
        back_image = Image.new('RGB', size, bg_color)
        text_mask = Image.new('L', back_image.size)

        # Draw the text on the text mask
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                             margins, line_spacing,
                                             position_x, position_y,
                                             align, justify,
                                             rotation_angle, rotation_options)

        # Composite the text image onto the background image using the rotated text mask
        image_out = Image.composite(text_image, back_image, rotated_text_mask)

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Text-Nodes#cr-draw-text"

        # Convert the PIL image back to a torch tensor
        return (pil2tensor(image_out), show_help,)

#---------------------------------------------------------------------------------------------------------------------#
class CR_MaskText:

    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        return {"required": {
                    "image": ("IMAGE",),
                    "text": ("STRING", {"multiline": True, "default": "text"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "background_color": (COLORS,),
                    "align": (ALIGN_OPTIONS,),
                    "justify": (JUSTIFY_OPTIONS,),
                    "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                    "rotation_options": (ROTATE_OPTIONS,),
                },
                "optional": {
                    "bg_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("IMAGE", "show_help", "text_image",)
    FUNCTION = "mask_text"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def mask_text(self, image, text, font_name, font_size,
              margins, line_spacing,
              position_x, position_y, background_color,
              align, justify,
              rotation_angle, rotation_options,
              bg_color_hex='#000000'):

        # Get RGB values for the background color
        bg_color = get_color_values(background_color, bg_color_hex, color_mapping)

        # Convert tensor images
        image_3d = image[0, :, :, :]
        text_image = tensor2pil(image_3d)
        background_image = Image.new('RGB', text_image.size, bg_color)
        text_mask = Image.new('L', text_image.size)

        # Draw the text on the text mask
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                            margins, line_spacing,
                                            position_x, position_y,
                                            align, justify,
                                            rotation_angle, rotation_options)

        # Invert the text mask
        inverted_mask = ImageOps.invert(rotated_text_mask)

        # Composite the text image onto the background image using the inverted text mask
        image_out = Image.composite(background_image, text_image, inverted_mask)

        # --- EXTRACT MINIMAL RGBA TEXT IMAGE ---
        bbox = inverted_mask.getbbox()
        if bbox:
            tx, ty, tw, th = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            cropped_mask = inverted_mask.crop(bbox)
            rgba_text_pil = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            rgba_draw = ImageDraw.Draw(rgba_text_pil)
            rgba_draw.bitmap((0, 0), cropped_mask, fill=(255, 255, 255))  # White text
        else:
            tx = ty = tw = th = 0
            rgba_text_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        text_tensor = pil2tensor(rgba_text_pil)
        text_hash = get_tensor_hash(text_tensor)

        # --- BUILD SCENE GRAPH ---
        bg_hash = get_tensor_hash(image)

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": text_image.width,
            "height": text_image.height,
            "images": [
                {
                    "x": 0, "y": 0, "width": text_image.width, "height": text_image.height,
                    "images": bg_hash
                },
                {
                    "x": tx, "y": ty, "width": tw, "height": th,
                    "images": text_hash
                }
            ]
        }

        show_help = json.dumps(show_help_data, indent=2)

        return (pil2tensor(image_out), show_help, text_tensor)

#---------------------------------------------------------------------------------------------------------------------#
class CR_CompositeText:

    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        return {"required": {
                    "image_text": ("IMAGE",),
                    "image_background": ("IMAGE",),
                    "text": ("STRING", {"multiline": True, "default": "text"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "align": (ALIGN_OPTIONS,),
                    "justify": (JUSTIFY_OPTIONS,),
                    "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                    "rotation_options": (ROTATE_OPTIONS,),
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE",)
    RETURN_NAMES = ("IMAGE", "show_help", "text_image",)
    FUNCTION = "composite_text"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def composite_text(self, image_text, image_background, text,
                   font_name, font_size,
                   margins, line_spacing,
                   position_x, position_y,
                   align, justify,
                   rotation_angle, rotation_options):

        # Convert tensor images
        image_text_3d = image_text[0, :, :, :]
        image_back_3d = image_background[0, :, :, :]

        # Create PIL images for the text and background layers and text mask
        text_image = tensor2pil(image_text_3d)
        back_image = tensor2pil(image_back_3d)
        text_mask = Image.new('L', back_image.size)

        # Draw the text on the text mask
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                            margins, line_spacing,
                                            position_x, position_y,
                                            align, justify,
                                            rotation_angle, rotation_options)

        # Composite the text image onto the background image using the rotated text mask
        image_out = Image.composite(text_image, back_image, rotated_text_mask)

        # --- EXTRACT MINIMAL RGBA TEXT IMAGE ---
        bbox = rotated_text_mask.getbbox()
        if bbox:
            tx, ty, tw, th = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            cropped_mask = rotated_text_mask.crop(bbox)
            rgba_text_pil = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            rgba_draw = ImageDraw.Draw(rgba_text_pil)
            rgba_draw.bitmap((0, 0), cropped_mask, fill=(255, 255, 255))  # Assume white
        else:
            tx = ty = tw = th = 0
            rgba_text_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        text_tensor = pil2tensor(rgba_text_pil)
        text_hash = get_tensor_hash(text_tensor)

        # --- BUILD SCENE GRAPH ---
        bg_hash = get_tensor_hash(image_background)

        # show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Text-Nodes#cr-composite-text"
        show_help_data = {
            "x": 0,
            "y": 0,
            "width": back_image.width,
            "height": back_image.height,
            "images": [
                { "x": 0, "y": 0, "width": back_image.width, "height": back_image.height, "images": bg_hash },
                { "x": tx, "y": ty, "width": tw, "height": th, "images": text_hash }
            ]
        }

        show_help = json.dumps(show_help_data, indent=2)

        return (pil2tensor(image_out), show_help, text_tensor)
#---------------------------------------------------------------------------------------------------------------------#
class CR_ArabicTextRTL:

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
                "arabic_text": ("STRING", {"multiline": True, "default": "شمس"}),
                }
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("arabic_text_rtl", "show help", )
    FUNCTION = "adjust_arabic_to_rtl"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def adjust_arabic_to_rtl(self, arabic_text):
        """
        Adjust Arabic text to read from right to left (RTL).

        Args:
            arabic_text (str): The Arabic text to be adjusted.

        Returns:
            str: The adjusted Arabic text in RTL format.
        """

        arabic_text_reshaped = arabic_reshaper.reshape(arabic_text)
        rtl_text = get_display(arabic_text_reshaped)

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Text-Nodes#cr-arabic-text-rtl"

        return (rtl_text, show_help,)

#---------------------------------------------------------------------------------------------------------------------#
class CR_SimpleTextWatermark:

    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        ALIGN_OPTIONS = ["center", "top left", "top center", "top right", "bottom left", "bottom center", "bottom right"]

        return {"required": {
                    "image": ("IMAGE",),
                    "text": ("STRING", {"multiline": False, "default": "@ your name"}),
                    "align": (ALIGN_OPTIONS,),
                    "opacity": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "font_color": (COLORS,),
                    "x_margin": ("INT", {"default": 20, "min": -1024, "max": 1024}),
                    "y_margin": ("INT", {"default": 20, "min": -1024, "max": 1024}),
                },
                "optional": {
                    "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "text_image")
    FUNCTION = "overlay_text"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def overlay_text(self, image, text, align,
                 font_name, font_size, font_color,
                 opacity, x_margin, y_margin, font_color_hex='#000000'):

        text_color = get_color_values(font_color, font_color_hex, color_mapping)
        total_images = []
        first_text_tensor = None  # Will capture minimal text image from first frame

        for i, img_tensor in enumerate(image):
            img_pil = tensor2pil(img_tensor)
            canvas_w, canvas_h = img_pil.size

            # Create transparent text layer
            textlayer = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(textlayer)

            # Load font
            font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", font_name)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()

            textsize = get_text_size(draw, text, font)

            # Compute position (same for all frames)
            if align == 'center':
                pos = [(canvas_w - textsize[0]) // 2, (canvas_h - textsize[1]) // 2]
            elif align == 'top left':
                pos = [x_margin, y_margin]
            elif align == 'top center':
                pos = [(canvas_w - textsize[0]) // 2, y_margin]
            elif align == 'top right':
                pos = [canvas_w - textsize[0] - x_margin, y_margin]
            elif align == 'bottom left':
                pos = [x_margin, canvas_h - textsize[1] - y_margin]
            elif align == 'bottom center':
                pos = [(canvas_w - textsize[0]) // 2, canvas_h - textsize[1] - y_margin]
            elif align == 'bottom right':
                pos = [canvas_w - textsize[0] - x_margin, canvas_h - textsize[1] - y_margin]

            draw.text(pos, text, font=font, fill=text_color)

            if opacity != 1:
                alpha = textlayer.split()[-1]
                alpha = alpha.point(lambda p: int(p * opacity))
                textlayer.putalpha(alpha)

            out_image = Image.composite(textlayer.convert("RGB"), img_pil, textlayer)
            total_images.append(pil2tensor(out_image))

            # Extract minimal RGBA text image from first frame only
            if first_text_tensor is None:
                bbox = textlayer.getbbox()
                if bbox:
                    cropped = textlayer.crop(bbox)
                    first_text_tensor = pil2tensor(cropped)
                else:
                    first_text_tensor = torch.zeros((1, 1, 1, 4), dtype=torch.float32)

        # === BUILD show_help ONCE ===
        bg_hash = get_tensor_hash(image[0].unsqueeze(0))  # Representative hash
        text_hash = get_tensor_hash(first_text_tensor)

        # Use position from last computed `pos` and `textsize`
        tx, ty = pos
        tw, th = textsize

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": canvas_w,
            "height": canvas_h,
            "images": [
                {
                    "x": 0,
                    "y": 0,
                    "width": canvas_w,
                    "height": canvas_h,
                    "images": bg_hash
                },
                {
                    "x": tx,
                    "y": ty,
                    "width": tw,
                    "height": th,
                    "images": text_hash
                }
            ]
        }

        show_help = json.dumps(show_help_data, indent=2)
        images_out = torch.cat(total_images, dim=0)

        return (images_out, show_help, first_text_tensor)

#---------------------------------------------------------------------------------------------------------------------#
class CR_SelectFont:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        if platform.system() == "Windows":
            system_root = os.environ.get("SystemRoot")
            font_dir = os.path.join(system_root, "Fonts") if system_root else None
       # Default debian-based Linux & MacOS font dirs
        elif platform.system() == "Linux":
            font_dir = "/usr/share/fonts/truetype"
        elif platform.system() == "Darwin":
            font_dir = "/System/Library/Fonts"

        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        return {"required": {
                "font_name": (file_list,),
                }
    }

    RETURN_TYPES = (any_type, "STRING",)
    RETURN_NAMES = ("font_name", "show_help",)
    FUNCTION = "select_font"
    CATEGORY = icons.get("Comfyroll/Graphics/Text")

    def select_font(self, font_name):

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Text-Nodes#cr-select-font"

        return (font_name, show_help,)

#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py
'''
NODE_CLASS_MAPPINGS = {
    "CR Overlay Text": CR_OverlayText,
    "CR Draw Text": CR_DrawText,
    "CR Mask Text": CR_MaskText,
    "CR Composite Text": CR_CompositeText,
    "CR Draw Perspective Text": CR_DrawPerspectiveText,
    "CR Arabic Text RTL": CR_ArabicTextRTL,
    "CR Simple Text Watermark": CR_SimpleTextWatermark,
    "CR Select Font": CR_SelectFont,
}
'''


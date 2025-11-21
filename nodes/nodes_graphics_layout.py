#---------------------------------------------------------------------------------------------------------------------#
# Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# for ComfyUI                                                 https://github.com/comfyanonymous/ComfyUI
#---------------------------------------------------------------------------------------------------------------------#

import json
import re
import numpy as np
import torch
import os
import hashlib
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageFilter
from ..categories import icons
from ..config import color_mapping, COLORS
from ..config import iso_sizes
from .functions_graphics import *
import threading
from server import PromptServer
from nodes import MAX_RESOLUTION
from comfy.utils import common_upscale, ProgressBar
from comfy import model_management
import torch.nn.functional as F

#---------------------------------------------------------------------------------------------------------------------#

ALIGN_OPTIONS = ["top", "center", "bottom"]
ROTATE_OPTIONS = ["text center", "image center"]
JUSTIFY_OPTIONS = ["left", "center", "right"]
PERSPECTIVE_OPTIONS = ["top", "bottom", "left", "right"]
#---------------------------------------------------------------------------------------------------------------------#

class PerRunCache:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.GLOBAL_SAVE_LAYOUTS = {}
        self.GLOBAL_SAVE_LEAF_DATA = {}

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def setLayout(self, key, value):
        # if leaf data already found, will not set layout assumed already resolve
        if self.GLOBAL_SAVE_LEAF_DATA.get(key) is not None:
            return
        self.GLOBAL_SAVE_LAYOUTS[key] = value

    def getLayout(self, key):
        return self.GLOBAL_SAVE_LAYOUTS.get(key, None)

    def getLayoutDict(self):
        return self.GLOBAL_SAVE_LAYOUTS

    def getLeafDataDict(self):
        return self.GLOBAL_SAVE_LEAF_DATA

    def setLeafData(self, key, value):
        if self.GLOBAL_SAVE_LEAF_DATA.get(key) is not None:
           raise ValueError(f"Leaf data for key {key} already set.")
        self.GLOBAL_SAVE_LEAF_DATA[key] = value
        # leaf data will always override container data
        if self.GLOBAL_SAVE_LAYOUTS.get(key) is not None:
            self.GLOBAL_SAVE_LAYOUTS.pop(key)

    def getLeafData(self, key):
        return self.GLOBAL_SAVE_LEAF_DATA.get(key, None)

    def clear(self):
        self.GLOBAL_SAVE_LAYOUTS.clear()
        self.GLOBAL_SAVE_LEAF_DATA.clear()


@PromptServer.instance.add_on_prompt_handler
def _(prompt):
    PerRunCache.get().clear()
    return prompt


#---------------------------------------------------------------------------------------------------------------------#
class CR_GetImageHash:

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
                "image": ("IMAGE",),
               }
    }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_hash"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def get_hash(self, image):

        image_hash = get_tensor_hash(image)

        return (f'"{image_hash}"', )


"""
example data_json nested tree layout structure
{  # eg. ComicPanelTemplateLayout or some other Container layout
 Width: 512,
 Height: 512,
 X: 15,
 Y: 15,
 Images: [{   # Images in comic panel / conatiner layout as list
       X: 5
       Y: 5
       Width: 32,
       Height: 32,
       Images: {  # nested injected (via string replace of sha256 image hash) of:
           # ComicPaneltemplateLayout or some other Container layout
           # where dimensions and positioning is to match parent width and height
                Width: 512,
                Height: 512,
                X: 10,
                Y: 10
                Images: [
                    {...},
                    ..
                ]
            }
   }, {...2nd image... }, {...3rd image...}, ..]
}
"""

class CR_SaveImageLeafData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE",),
                "json_data_str": ("STRING",),
               }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # for now always return true to force no caching, CONSIDER: validate against PerRunCache if invalidated
        return float("NaN")

    def execute(self, image, json_data_str):
        try:
            json.loads(json_data_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in json_data_str.")
        image_hash = f'"{get_tensor_hash(image)}"'
        PerRunCache.get().setLeafData(image_hash, json_data_str)
        print(f"Set Leaf Data for image hash: {image_hash}: {json_data_str}")
        return ()


class CR_SaveImageHashLayout:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE",),
                "json_node_str": ("STRING",),
               }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # for now always return true to force no caching, CONSIDER: validate against PerRunCache if invalidated
        return float("NaN")

    def execute(self, image, json_node_str):
        try:
            json.loads(json_node_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in json_node_str.")
        image_hash = f'"{get_tensor_hash(image)}"'
        PerRunCache.get().setLayout(image_hash, json_node_str)
        print(f"Set Layout for image hash: {image_hash}: {json_node_str}")
        return ()

"""
Converts a JSON layout tree structure into a flattened list of image regions with their global coordinates and dimensions.
"""
class CR_FlattenedLayoutRegionsJSON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data_json": ("STRING", {"multiline": True, "default": '{}'}),
            },
            "optional": {
                "region_keys": ("STRING", {"multiline": True, "default": ""}),
                "preview_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", )
    RETURN_NAMES = ("STRING", "preview_image", )
    FUNCTION = "get_layout_regions"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # for now always return true to force no caching, CONSIDER: validate against PerRunCache if invalidated
        return float("NaN")

    def get_layout_regions(self, data_json, region_keys=None, preview_image=None):
        regions = []

        region_keys = region_keys.strip()
        if region_keys: # filter hash by region keys
            region_keys = set(region_keys.strip().splitlines()) if not isinstance(region_keys, list) else set(region_keys)
        else: # no filtering by region keys
            region_keys = None

        GLOBAL_SAVE_LAYOUTS = PerRunCache.get().getLayoutDict()
        GLOBAL_SAVE_LEAF_DATA = PerRunCache.get().getLeafDataDict()

        # print("Checking layout region replacements...")
        # print(GLOBAL_SAVE_LAYOUTS)
        # print("Checking leaf data replacements...")
        # print(GLOBAL_SAVE_LEAF_DATA)

        data_json = data_json.strip()

        replaced = True
        while replaced:
            replaced = False
            for key, value in GLOBAL_SAVE_LAYOUTS.items():
                if key in data_json:
                    # Replace all occurrences of the key with its value (as string)
                    data_json = data_json.replace(key, value)
                    replaced = True

        try:
            data_json = json.loads(data_json)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in data_json. \n" + data_json)

        if not isinstance(data_json, dict):
            raise ValueError("data_json must be a JSON object.")

        # iterative depth first search recurse to collect regions
        stack = [data_json]

        # 0,1: accumiulated global x and y offset positions for current item in stack,
        # 2,3: globally-projected width and height value for current item in stack
        stack_origins = [(0, 0, -1, -1)]

        while stack:
            current = stack.pop()
            origin = stack_origins.pop()

            # local width and height values
            width = current.get("width", -1)
            height = current.get("height", -1)
            # localToGlobal scale x and  scale y values to be determined
            scalex = 1
            scaley = 1
            # determine localToGlobal scales
            # convert local width and height values to global width and height values if needed
            if origin[2] >= 0 and width >= 0 and origin[2] != width :
                scalex = origin[2] / width
                width *= scalex
            if origin[3] >= 0 and height >= 0 and origin[3] != height:
                scaley = origin[3] / height
                height *= scaley

            # global x and y coordinate (top-left)
            cx = current.get("x", 0)*scalex + origin[0]
            cy = current.get("y", 0)*scaley + origin[1]

            images = current.get("images")
            if images:
                if isinstance(images, list):
                    for image in reversed(images):
                        if not isinstance(image, dict):
                            raise ValueError("Each image in 'images' list array must be a dictionary")
                        stack.append(image)
                        iwidth = image.get("width", -1)
                        iheight = image.get("height", -1)
                        stack_origins.append((
                            cx, cy,
                            iwidth*scalex if iwidth >= 0 else -1,
                            iheight*scaley if iheight >= 0 else -1
                            )
                        )

                elif isinstance(images, dict): # scaled nested region within current space
                    stack.append(images)
                    stack_origins.append((cx, cy, width, height))
                else: # assumed string leaf sha256 hash of final image
                    if region_keys is None or images in region_keys:
                        regions.append({
                            "x": cx,
                            "y": cy,
                            "width": width,
                            "height": height,
                            "data": images
                        })

        if preview_image is not None:
            # display red outlines for all the regions in preview_image
            preview_image = tensor2pil(preview_image)
            draw = ImageDraw.Draw(preview_image)
            for region in regions:
                x = region["x"]
                y = region["y"]
                w = region["width"]
                h = region["height"]
                random_outline_color = tuple(np.random.randint(0, 256, size=3).tolist())
                draw.rectangle([x, y, x + w, y + h], outline=random_outline_color, width=3)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
            preview_image = pil2tensor(preview_image)
        else:
            # dummy 1x1 preview image
            preview_image = Image.new('RGB', (1, 1), color=(0, 0, 0))
            preview_image = pil2tensor(preview_image)

        json_array_regions_str = json.dumps(regions)
        replaced = True
        while replaced:
            replaced = False
            for key, value in GLOBAL_SAVE_LEAF_DATA.items():
                if key in json_array_regions_str:
                    json_array_regions_str = json_array_regions_str.replace(key, value)
                    replaced = True

        return (json_array_regions_str, preview_image,)


class CR_PageLayout:

    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        layout_options = ["header", "footer", "header and footer", "no header or footer"]

        return {"required": {
                "layout_options": (layout_options,),
                "image_panel": ("IMAGE",),
                "header_height": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "header_text": ("STRING", {"multiline": True, "default": "text"}),
                "header_align": (JUSTIFY_OPTIONS, ),
                "footer_height": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "footer_text": ("STRING", {"multiline": True, "default": "text"}),
                "footer_align": (JUSTIFY_OPTIONS, ),
                "font_name": (file_list,),
                "font_color": (COLORS,),
                "header_font_size": ("INT", {"default": 150, "min": 0, "max": 1024}),
                "footer_font_size": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "border_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "border_color": (COLORS,),
                "background_color": (COLORS,),
               },
                "optional": {
                "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "border_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "bg_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                'layout_alignment': (["vertical", "horizontal"],  {"default": "vertical"}),
                "header_image": ("IMAGE",),
                "footer_image": ("IMAGE",),
                "margins": ("INT", {"default": 50, "min": 0, "max": 1024}),
               }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "layout"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def layout(self, layout_options, image_panel,
               border_thickness, border_color, background_color,
               header_height, header_text, header_align,
               footer_height, footer_text, footer_align,
               font_name, font_color,
               header_font_size, footer_font_size,
               font_color_hex='#000000', border_color_hex='#000000', bg_color_hex='#000000', layout_alignment='horizontal', header_image=None, footer_image=None, margins=50):

        # Get RGB values for the text and background colors
        font_color = get_color_values(font_color, font_color_hex, color_mapping)
        border_color = get_color_values(border_color, border_color_hex, color_mapping)
        bg_color = get_color_values(background_color, bg_color_hex, color_mapping)

        main_panel_hash = get_tensor_hash(image_panel)
        main_panel = tensor2pil(image_panel)

        if header_image is not None:
            header_image_pil = tensor2pil(header_image)
        else:
            header_image_pil = None


        if footer_image is not None:
            footer_image_pil = tensor2pil(footer_image)
        else:
            footer_image_pil = None

        # Get image width and height
        image_width = main_panel.width
        image_height = main_panel.height

        # Set defaults
        line_spacing = 0
        position_x = 0
        position_y = 0
        align = "center"
        rotation_angle = 0
        rotation_options = "image center"
        font_outline_thickness = 0
        font_outline_color = "black"

        images = []

        yx_offset = 0
        is_vertical = layout_alignment == "vertical"

        ### Create text panels and add to images array
        if layout_options == "header" or layout_options == "header and footer":
            header_panel = text_panel(image_width, header_height, header_text,
                                      font_name, header_font_size, font_color,
                                      font_outline_thickness, font_outline_color,
                                      bg_color,
                                      margins, line_spacing,
                                      position_x, position_y,
                                      align, header_align,
                                      rotation_angle, rotation_options, background_image=header_image_pil)
            images.append(header_panel)
            yx_offset += header_panel.height if is_vertical else header_panel.width

        images.append(main_panel)

        yx_offset_footer = (main_panel.height if is_vertical else main_panel.width) + yx_offset

        if layout_options == "footer" or layout_options == "header and footer":
            footer_panel = text_panel(image_width, footer_height, footer_text,
                                      font_name, footer_font_size, font_color,
                                      font_outline_thickness, font_outline_color,
                                      bg_color,
                                      margins, line_spacing,
                                      position_x, position_y,
                                      align, footer_align,
                                      rotation_angle, rotation_options,
                                      background_image=footer_image_pil
                                      )
            images.append(footer_panel)

        combined_image = combine_images(images, layout_alignment, bg_color)

        # Add a border to the combined image
        if border_thickness > 0:
            combined_image = ImageOps.expand(combined_image, border_thickness, border_color)

        offset_padding = border_thickness

        images_list = [{ # main_panel image
            "x": offset_padding if is_vertical else yx_offset + offset_padding,
            "y": yx_offset + offset_padding if is_vertical else offset_padding,
            "width": image_width,
            "height": image_height,
            "images": main_panel_hash
        }]
        if header_image is not None and layout_options in ("header", "header and footer"):
            images_list.insert(0, { # header image
                "x": offset_padding,
                "y": offset_padding,
                "width": header_image_pil.width,
                "height": header_image_pil.height,
                "images": get_tensor_hash(header_image)
            })
        if footer_image is not None and layout_options in ("footer", "header and footer"):
            images_list.append({ # footer image
                "x": offset_padding if is_vertical else yx_offset_footer + offset_padding,
                "y": yx_offset_footer + offset_padding if is_vertical else offset_padding,
                "width": footer_image_pil.width,
                "height": footer_image_pil.height,
                "images": get_tensor_hash(footer_image)
            })


        show_help = json.dumps({
            "width": combined_image.width,
            "height": combined_image.height,
            "images": images_list
        })
        #show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-page-layout"

        return (pil2tensor(combined_image), show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_SimpleTitles:

    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        layout_options = ["header", "footer", "header and footer", "no header or footer"]

        return {"required": {
                "image": ("IMAGE",),
                "header_text": ("STRING", {"multiline": True, "default": "text"}),
                "header_height": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "header_font_size": ("INT", {"default": 150, "min": 0, "max": 1024}),
                "header_align": (JUSTIFY_OPTIONS, ),
                "footer_text": ("STRING", {"multiline": True, "default": "text"}),
                "footer_height": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "footer_font_size": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "footer_align": (JUSTIFY_OPTIONS, ),
                "font_name": (file_list,),
                "font_color": (COLORS,),
                "background_color": (COLORS,),
               },
                "optional": {
                "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "bg_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "margins": ("INT", {"default": 50, "min": 0, "max": 1024}),
               }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "layout"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def layout(self, image,
               header_height, header_text, header_align, header_font_size,
               footer_height, footer_text, footer_align, footer_font_size,
               font_name, font_color, background_color,
               font_color_hex='#000000', bg_color_hex='#000000', margins=50):

        # Get RGB values for the text and background colors
        font_color = get_color_values(font_color, font_color_hex, color_mapping)
        bg_color = get_color_values(background_color, bg_color_hex, color_mapping)

        main_panel = tensor2pil(image)
        # Get image width and height
        image_width = main_panel.width
        image_height = main_panel.height

        # Set defaults
        line_spacing = 0
        position_x = 0
        position_y = 0
        align = "center"
        rotation_angle = 0
        rotation_options = "image center"
        font_outline_thickness = 0
        font_outline_color = "black"
        images = []
        ### Create text panels and add to images array
        if header_height >0:
            header_panel = text_panel(image_width, header_height, header_text,
                                      font_name, header_font_size, font_color,
                                      font_outline_thickness, font_outline_color,
                                      bg_color,
                                      margins, line_spacing,
                                      position_x, position_y,
                                      align, header_align,
                                      rotation_angle, rotation_options)
            images.append(header_panel)

        images.append(main_panel)

        if footer_height >0:
            footer_panel = text_panel(image_width, footer_height, footer_text,
                                      font_name, footer_font_size, font_color,
                                      font_outline_thickness, font_outline_color,
                                      bg_color,
                                      margins, line_spacing,
                                      position_x, position_y,
                                      align, footer_align,
                                      rotation_angle, rotation_options)
            images.append(footer_panel)

        combined_image = combine_images(images, 'vertical')

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-simple_titles"

        return (pil2tensor(combined_image), show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_ImagePanel:

    @classmethod
    def INPUT_TYPES(s):

        directions = ["horizontal", "vertical"]

        return {"required": {
                "image_1": ("IMAGE",),
                "border_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "border_color": (COLORS,),
                "outline_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "outline_color": (COLORS[1:],),
                "layout_direction": (directions,),
               },
                "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "border_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
               }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, image_1,
                   border_thickness, border_color,
                   outline_thickness, outline_color,
                   layout_direction, image_2=None, image_3=None, image_4=None,
                   border_color_hex='#000000'):

        border_color = get_color_values(border_color, border_color_hex, color_mapping)

        image_tensor_hash_list = []

        # Convert PIL images to NumPy arrays
        images = []
        #image_1 = image_1[0, :, :, :]
        images.append(tensor2pil(image_1))
        image_tensor_hash_list.append(image_1)
        if image_2 is not None:
            #image_2 = image_2[0, :, :, :]
            images.append(tensor2pil(image_2))
            image_tensor_hash_list.append(image_2)
        if image_3 is not None:
            #image_3 = image_3[0, :, :, :]
            images.append(tensor2pil(image_3))
            image_tensor_hash_list.append(image_3)
        if image_4 is not None:
            #image_4 = image_4[0, :, :, :]
            images.append(tensor2pil(image_4))
            image_tensor_hash_list.append(image_4)
        # Apply borders and outlines to each image
        images = apply_outline_and_border(images, outline_thickness, outline_color, border_thickness, border_color)

        image_tensor_hash_list = [get_tensor_hash(image) for image in image_tensor_hash_list]

        combined_image, combined_images_positions = combine_images2(images, layout_direction)
        offset_padding = border_thickness + outline_thickness

        show_help = json.dumps({
            "width": combined_image.width,
            "height": combined_image.height,
            "images": [
                {
                    "x": coord[0] + offset_padding,
                    "y": coord[1] + offset_padding,
                    "width": images[idx].width - 2 * offset_padding,
                    "height": images[idx].height - 2 * offset_padding,
                    "images": image_tensor_hash_list[idx],
                } for idx, coord in enumerate(combined_images_positions)
            ]
        })
        # show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-image-panel"

        return (pil2tensor(combined_image), show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_ImageListPanel:
    @classmethod
    def INPUT_TYPES(s):
        directions = ["horizontal", "vertical"]
        return {
            "required": {
                "images": ("IMAGE",),
                "border_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "border_color": (COLORS,),
                "outline_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "outline_color": (COLORS[1:],),
                "layout_direction": (directions,),
            },
            "optional": {
                "border_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "show_help")
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, images, border_thickness, border_color, outline_thickness, outline_color, layout_direction, border_color_hex='#000000'):

        border_color = get_color_values(border_color, border_color_hex, color_mapping)
        pil_images = [tensor2pil(img) for img in images]
        image_tensor_hash_list = [get_tensor_hash(img) for img in images]
        pil_images = apply_outline_and_border(
            pil_images, outline_thickness, outline_color, border_thickness, border_color
        )
        combined_image, combined_images_positions = combine_images2(pil_images, layout_direction)
        offset_padding = border_thickness + outline_thickness

        # Generate show_help metadata
        show_help = json.dumps({
            "width": combined_image.width,
            "height": combined_image.height,
            "images": [
                {
                    "x": coord[0] + offset_padding,
                    "y": coord[1] + offset_padding,
                    "width": pil_images[idx].width - 2 * offset_padding,
                    "height": pil_images[idx].height - 2 * offset_padding,
                    "hash": image_tensor_hash_list[idx],
                }
                for idx, coord in enumerate(combined_images_positions)
            ]
        })
        # Return as standard ComfyUI image tensor
        return (pil2tensor(combined_image), show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_ImageGridPanel:

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
                    "images": ("IMAGE",),
                    "border_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "border_color": (COLORS,),
                    "outline_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "outline_color": (COLORS[1:],),
                    "max_columns": ("INT", {"default": 5, "min": 0, "max": 256}),
                },
                "optional": {
                    "border_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, images,
                   border_thickness, border_color,
                   outline_thickness, outline_color,
                   max_columns, border_color_hex='#000000'):

        border_color = get_color_values(border_color, border_color_hex, color_mapping)

        image_tensor_hash_list = [get_tensor_hash(image) for image in images]

        # Convert PIL images
        images = [tensor2pil(image) for image in images]

        # Apply borders and outlines to each image
        images = apply_outline_and_border(images, outline_thickness, outline_color, border_thickness, border_color)

        combined_image, combined_images_positions = make_grid_panel2(images, max_columns)
        offset_padding = border_thickness + outline_thickness

        image_out = pil2tensor(combined_image)

        show_help = json.dumps({
            "width": combined_image.width,
            "height": combined_image.height,
            "images": [
                {
                    "x": coord[0] + offset_padding,
                    "y": coord[1] + offset_padding,
                    "width": images[idx].width - 2 * offset_padding,
                    "height": images[idx].height - 2 * offset_padding,
                    "images": image_tensor_hash_list[idx],
                } for idx, coord in enumerate(combined_images_positions)
            ]
        })
        # show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-image-grid-panel"

        return (image_out, show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_ImageBorder:

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
                    "image": ("IMAGE",),
                    "top_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "bottom_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "left_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "right_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "border_color": (COLORS,),
                    "outline_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                    "outline_color": (COLORS[1:],),
                },
                "optional": {
                    "border_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, image,
                   top_thickness, bottom_thickness,
                   left_thickness, right_thickness, border_color,
                   outline_thickness, outline_color,
                   border_color_hex='#000000'):

        images = []

        border_color = get_color_values(border_color, border_color_hex, color_mapping)

        for img in image:
            img = tensor2pil(img)

            # Apply the outline
            if outline_thickness > 0:
                img = ImageOps.expand(img, outline_thickness, fill=outline_color)

            # Apply the borders
            if left_thickness > 0 or right_thickness > 0 or top_thickness > 0 or bottom_thickness > 0:
                img = ImageOps.expand(img, (left_thickness, top_thickness, right_thickness, bottom_thickness), fill=border_color)

            images.append(pil2tensor(img))

        images = torch.cat(images, dim=0)

        # --- BUILD SCENE GRAPH (minimal addition) ---
        if len(image) == 0:
            show_help = json.dumps({"x":0,"y":0,"width":0,"height":0})
        else:
            bg_hash = get_tensor_hash(image[0].unsqueeze(0))
            pil_first = tensor2pil(image[0])
            orig_w, orig_h = pil_first.size

            final_w = orig_w + left_thickness + right_thickness + 2 * outline_thickness
            final_h = orig_h + top_thickness + bottom_thickness + 2 * outline_thickness

            show_help_data = {
                "x": 0,
                "y": 0,
                "width": final_w,
                "height": final_h,
                "images": [
                    {
                        "x": left_thickness + outline_thickness,
                        "y": top_thickness + outline_thickness,
                        "width": orig_w,
                        "height": orig_h,
                        "images": bg_hash
                    }
                ]
            }
            show_help = json.dumps(show_help_data)

        # Comment out the old help URL
        # show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-image-border"

        return (images, show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_ColorPanel:

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
                    "panel_width": ("INT", {"default": 512, "min": 8, "max": 4096}),
                    "panel_height": ("INT", {"default": 512, "min": 8, "max": 4096}),
                    "fill_color": (COLORS,),
                },
                "optional": {
                    "fill_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, panel_width, panel_height,
                   fill_color, fill_color_hex='#000000'):

        fill_color = get_color_values(fill_color, fill_color_hex, color_mapping)

        size = (panel_width, panel_height)
        panel = Image.new('RGB', size, fill_color)

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-color-panel"

        return (pil2tensor(panel), show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_SimpleTextPanel:

    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and (f.lower().endswith(".ttf") or f.lower().endswith(".otf"))]

        return {
            "required": {
                "panel_width": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "panel_height": ("INT", {"default": 512, "min": 0, "max": 4096}),
                "text": ("STRING", {"multiline": True, "default": "text"}),
                "font_name": (file_list,),
                "font_color": (COLORS,),
                "font_size": ("INT", {"default": 100, "min": 0, "max": 1024}),
                "font_outline_thickness": ("INT", {"default": 0, "min": 0, "max": 50}),
                "font_outline_color": (COLORS,),
                "background_color": (COLORS,),
                "align": (ALIGN_OPTIONS,),
                "justify": (JUSTIFY_OPTIONS,),
                "wrap": ("BOOLEAN", {"default": False}),
                "margins": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
            },
            "optional": {
                "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                "bg_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "show_help")
    FUNCTION = "layout"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def layout(self, panel_width, panel_height,
               text, align, justify,
               font_name, font_color, font_size,
               font_outline_thickness, font_outline_color,
               background_color,
               wrap=False,
               margins=50,
               line_spacing=0,
               font_color_hex='#000000',
               font_outline_color_hex='#000000',
               bg_color_hex='#000000'):

        # Get RGB values for colors
        font_color = get_color_values(font_color, font_color_hex, color_mapping)
        outline_color = get_color_values(font_outline_color, font_outline_color_hex, color_mapping)
        bg_color = get_color_values(background_color, bg_color_hex, color_mapping)

        # --- AUTO WORD WRAPPING & SIZE ESTIMATION ---
        font_path = os.path.join("fonts", font_name)
        resolved_font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), font_path)
        try:
            font = ImageFont.truetype(resolved_font_path, font_size)
        except:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        # Process text (with optional wrapping)
        if wrap:
            words = text.split(' ')
            lines = []
            line = ""
            available_width = max(1, panel_width - 2 * margins)  # Avoid zero/negative

            for word in words:
                test_line = f"{line} {word}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
                if text_width <= available_width or not line:
                    line = test_line
                else:
                    if line:
                        lines.append(line)
                        line = word
                    else:
                        lines.append(word)
            if line:
                lines.append(line)
            processed_text = "\n".join(lines)
        else:
            processed_text = text
            lines = processed_text.split('\n')

        # Auto-calculate height if requested
        if panel_height == 0:
            total_text_height = 0
            max_line_width = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_h = bbox[3] - bbox[1] + line_spacing
                line_w = bbox[2] - bbox[0]
                total_text_height += line_h
                max_line_width = max(max_line_width, line_w)

            # Add margins and ensure minimum size
            auto_height = int(total_text_height + 2 * margins)
            panel_height = max(1, auto_height)  # Avoid zero/negative

        panel_width = max(1, panel_width) if panel_width > 0 else 1

        # Clamp margins to safe bounds
        margins = min(margins, panel_height // 2, panel_width // 2)

        # --- CREATE TEXT PANEL ---
        panel = text_panel(
            image_width=panel_width,
            image_height=panel_height,
            text=processed_text,
            font_name=font_name,
            font_size=font_size,
            font_color=font_color,
            font_outline_thickness=font_outline_thickness,
            font_outline_color=outline_color,
            background_color=bg_color,
            margins=margins,
            line_spacing=line_spacing,
            position_x=0,
            position_y=0,
            align=align,
            justify=justify,
            rotation_angle=0,
            rotation_options="image center"
        )

        # Convert to tensor
        result = pil2tensor(panel)
        output_hash = get_tensor_hash(result)

        # --- BUILD SCENE GRAPH ---
        content_x = margins
        content_y = margins
        content_w = max(0, panel_width - 2 * margins)
        content_h = max(0, panel_height - 2 * margins)

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": panel_width,
            "height": panel_height,
            "images": output_hash,
            "text": {
                "x": int(content_x),
                "y": int(content_y),
                "width": int(content_w),
                "height": int(content_h),
                "content": text,
            }
        }

        # Comment out old URL
        # show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-simple-text-panel"
        show_help = json.dumps(show_help_data)

        return (result, show_help)

#---------------------------------------------------------------------------------------------------------------------#
class CR_OverlayTransparentImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "back_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "transparency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "overlay_scale_factor": ("FLOAT", {"default": 1.000, "min": 0.000, "max": 100.000, "step": 0.001}),
                "anchor": ([
                    "center",
                    "top", "bottom", "left", "right",
                    "top-left", "top-right", "bottom-left", "bottom-right"
                ], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "overlay_rgba")
    FUNCTION = "overlay_image"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def overlay_image(self, back_image, overlay_image,
                      transparency, offset_x, offset_y, rotation_angle, overlay_scale_factor=1.0,
                      anchor="center"):

        # Convert tensors to PIL
        bg_pil = tensor2pil(back_image)
        fg_pil = tensor2pil(overlay_image)

        bg_w, bg_h = bg_pil.size

        # --- Handle Alpha: Combine original alpha with user transparency ---
        if fg_pil.mode in ('RGBA', 'LA') or (fg_pil.mode == 'P' and 'transparency' in fg_pil.info):
            if fg_pil.mode == 'RGBA':
                r, g, b, orig_alpha = fg_pil.split()
            elif fg_pil.mode == 'LA':
                l, orig_alpha = fg_pil.split()
            else:
                orig_alpha = fg_pil.convert('RGBA').split()[-1]

            user_alpha_factor = 1.0 - transparency
            final_alpha_array = np.array(orig_alpha, dtype=np.float32) * user_alpha_factor
            final_alpha_array = np.clip(final_alpha_array, 0, 255).astype(np.uint8)
            final_alpha = Image.fromarray(final_alpha_array, mode='L')

            fg_pil = Image.merge('RGBA', (*fg_pil.convert('RGB').split(), final_alpha))
        else:
            fg_pil = fg_pil.convert('RGBA')
            alpha_value = int(255 * (1 - transparency))
            fg_pil.putalpha(Image.new('L', fg_pil.size, alpha_value))

        # --- ROTATE around center ---
        if rotation_angle != 0.0:
            fg_pil = fg_pil.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

        # --- SCALE ---
        new_size = (int(fg_pil.width * overlay_scale_factor), int(fg_pil.height * overlay_scale_factor))
        fg_pil = fg_pil.resize(new_size, Image.LANCZOS)
        rotated_scaled_w, rotated_scaled_h = fg_pil.size

        # --- POSITION using anchor ---
        anchor = anchor.lower()

        if anchor == "center":
            x = (bg_w - rotated_scaled_w) // 2 + offset_x
            y = (bg_h - rotated_scaled_h) // 2 + offset_y
        elif anchor == "top":
            x = (bg_w - rotated_scaled_w) // 2 + offset_x
            y = 0 + offset_y
        elif anchor == "bottom":
            x = (bg_w - rotated_scaled_w) // 2 + offset_x
            y = bg_h - rotated_scaled_h + offset_y
        elif anchor == "left":
            x = 0 + offset_x
            y = (bg_h - rotated_scaled_h) // 2 + offset_y
        elif anchor == "right":
            x = bg_w - rotated_scaled_w + offset_x
            y = (bg_h - rotated_scaled_h) // 2 + offset_y
        elif anchor == "top-left":
            x = 0 + offset_x
            y = 0 + offset_y
        elif anchor == "top-right":
            x = bg_w - rotated_scaled_w + offset_x
            y = 0 + offset_y
        elif anchor == "bottom-left":
            x = 0 + offset_x
            y = bg_h - rotated_scaled_h + offset_y
        elif anchor == "bottom-right":
            x = bg_w - rotated_scaled_w + offset_x
            y = bg_h - rotated_scaled_h + offset_y
        else:
            x = (bg_w - rotated_scaled_w) // 2
            y = (bg_h - rotated_scaled_h) // 2

        # --- COMPOSITE ---
        result = bg_pil.copy()
        result = result.convert("RGBA")
        result.paste(fg_pil, (int(x), int(y)), mask=fg_pil)
        result = result.convert("RGB")  # Back to RGB

        # --- OUTPUT: Final processed overlay (RGBA) ---
        overlay_rgba_tensor = pil2tensor(fg_pil)

        # --- BUILD SCENE GRAPH ---
        bg_hash = get_tensor_hash(back_image)
        fg_hash = get_tensor_hash(overlay_rgba_tensor)  # Based on final RGBA

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": bg_w,
            "height": bg_h,
            "images": [
                {
                    "x": 0,
                    "y": 0,
                    "width": bg_w,
                    "height": bg_h,
                    "images": bg_hash
                },
                {
                    "x": int(x),
                    "y": int(y),
                    "width": rotated_scaled_w,
                    "height": rotated_scaled_h,
                    "images": fg_hash
                }
            ]
        }

        # Comment out old URL
        # show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-overlay-transparent-image"
        show_help = json.dumps(show_help_data)

        return (pil2tensor(result), show_help, overlay_rgba_tensor)

#---------------------------------------------------------------------------------------------------------------------#
class CR_ExpandedOverlayTransparentImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "back_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "transparency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "overlay_scale_factor": ("FLOAT", {"default": 1.000, "min": 0.000, "max": 100.000, "step": 0.001}),
                "anchor": ([
                    "center",
                    "top", "bottom", "left", "right",
                    "top-left", "top-right", "bottom-left", "bottom-right"
                ], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE",)
    RETURN_NAMES = ("rgba_canvas", "show_help", "overlay_rgba",)
    FUNCTION = "composite"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def composite(self, back_image, overlay_image,
                  transparency, offset_x, offset_y, rotation_angle, overlay_scale_factor=1.0,
                  anchor="center"):

        # Convert tensors to PIL
        bg_pil = tensor2pil(back_image[0]).convert("RGBA")  # Use first image
        fg_pil = tensor2pil(overlay_image[0]).convert("RGBA")  # Use first image

        bg_w, bg_h = bg_pil.size

        # --- Handle Alpha: Combine original alpha with user transparency ---
        r, g, b, orig_alpha = fg_pil.split()
        user_alpha_factor = 1.0 - transparency
        final_alpha_array = np.array(orig_alpha, dtype=np.float32) * user_alpha_factor
        final_alpha_array = np.clip(final_alpha_array, 0, 255).astype(np.uint8)
        final_alpha = Image.fromarray(final_alpha_array, mode='L')
        fg_pil = Image.merge('RGBA', (r, g, b, final_alpha))

        # --- ROTATE ---
        if rotation_angle != 0.0:
            fg_pil = fg_pil.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

        # --- SCALE ---
        new_size = (int(fg_pil.width * overlay_scale_factor), int(fg_pil.height * overlay_scale_factor))
        fg_pil = fg_pil.resize(new_size, Image.LANCZOS)
        fg_w, fg_h = fg_pil.size

        # --- POSITION using anchor (relative to center of bg) ---
        cx = bg_w // 2
        cy = bg_h // 2

        anchor = anchor.lower()

        if anchor == "center":
            fx = cx - fg_w // 2 + offset_x
            fy = cy - fg_h // 2 + offset_y
        elif anchor == "top":
            fx = cx - fg_w // 2 + offset_x
            fy = 0 + offset_y
        elif anchor == "bottom":
            fx = cx - fg_w // 2 + offset_x
            fy = bg_h - fg_h + offset_y
        elif anchor == "left":
            fx = 0 + offset_x
            fy = cy - fg_h // 2 + offset_y
        elif anchor == "right":
            fx = bg_w - fg_w + offset_x
            fy = cy - fg_h // 2 + offset_y
        elif anchor == "top-left":
            fx = 0 + offset_x
            fy = 0 + offset_y
        elif anchor == "top-right":
            fx = bg_w - fg_w + offset_x
            fy = 0 + offset_y
        elif anchor == "bottom-left":
            fx = 0 + offset_x
            fy = bg_h - fg_h + offset_y
        elif anchor == "bottom-right":
            fx = bg_w - fg_w + offset_x
            fy = bg_h - fg_h + offset_y
        else:
            fx = cx - fg_w // 2 + offset_x
            fy = cy - fg_h // 2 + offset_y

        # --- DETERMINE FINAL CANVAS SIZE ---
        # Include both background and overlay bounds
        min_x = min(0, fx)
        min_y = min(0, fy)
        max_x = max(bg_w, fx + fg_w)
        max_y = max(bg_h, fy + fg_h)

        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)

        # Offset for placing background
        bg_x = -min_x
        bg_y = -min_y

        # Final overlay position on canvas
        fg_canvas_x = fx - min_x
        fg_canvas_y = fy - min_y

        # --- CREATE FINAL RGBA CANVAS ---
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        # Paste background
        canvas.paste(bg_pil, (bg_x, bg_y), mask=bg_pil)

        # Paste overlay
        canvas.paste(fg_pil, (fg_canvas_x, fg_canvas_y), mask=fg_pil)

        overlay_rgba_tensor = pil2tensor(fg_pil)

        # --- BUILD SCENE GRAPH ---
        bg_hash = get_tensor_hash(back_image)
        fg_hash = get_tensor_hash(overlay_rgba_tensor)

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": canvas_w,
            "height": canvas_h,
            "images": [
                {
                    "x": bg_x,
                    "y": bg_y,
                    "width": bg_w,
                    "height": bg_h,
                    "images": bg_hash
                },
                {
                    "x": fg_canvas_x,
                    "y": fg_canvas_y,
                    "width": fg_w,
                    "height": fg_h,
                    "images": fg_hash
                }
            ]
        }

        show_help = json.dumps(show_help_data)

        return (pil2tensor(canvas), show_help, overlay_rgba_tensor)
#---------------------------------------------------------------------------------------------------------------------#
class CR_FeatheredBorder:

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
                    "image": ("IMAGE",),
                    "top_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "bottom_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "left_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "right_thickness": ("INT", {"default": 0, "min": 0, "max": 4096}),
                    "border_color": (COLORS,),
                    "feather_amount": ("INT", {"default": 0, "min": 0, "max": 1024}),
                },
                "optional": {
                    "border_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_border"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_border(self, image,
                   top_thickness, bottom_thickness,
                   left_thickness, right_thickness, border_color,
                   feather_amount,
                   border_color_hex='#000000'):

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-feathered-border"

        images = []

        border_color = get_color_values(border_color, border_color_hex, color_mapping)

        for img in image:
            im = tensor2pil(img)

            RADIUS = feather_amount

            # Paste image on white background
            diam = 2*RADIUS
            back = Image.new('RGB', (im.size[0]+diam, im.size[1]+diam), border_color)
            back.paste(im, (RADIUS, RADIUS))

            # Create paste mask
            mask = Image.new('L', back.size, 0)
            draw = ImageDraw.Draw(mask)
            x0, y0 = 0, 0
            x1, y1 = back.size
            for d in range(diam+RADIUS):
                x1, y1 = x1-1, y1-1
                alpha = 255 if d<RADIUS else int(255*(diam+RADIUS-d)/diam)
                draw.rectangle([x0, y0, x1, y1], outline=alpha)
                x0, y0 = x0+1, y0+1

            # Blur image and paste blurred edge according to mask
            blur = back.filter(ImageFilter.GaussianBlur(RADIUS/2))
            back.paste(blur, mask=mask)

            # Apply the borders
            if left_thickness > 0 or right_thickness > 0 or top_thickness > 0 or bottom_thickness > 0:
                img = ImageOps.expand(back, (left_thickness, top_thickness, right_thickness, bottom_thickness), fill=border_color)
            else:
                img = back

            images.append(pil2tensor(img))

        images = torch.cat(images, dim=0)

        return (images, show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_HalfDropPanel:

    @classmethod
    def INPUT_TYPES(s):

        patterns = ["none", "half drop", "quarter drop", "custom drop %"]

        return {"required": {
                    "image": ("IMAGE",),
                    "pattern": (patterns,),
                },
                "optional": {
                    "drop_percentage": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                }
    }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, image, pattern, drop_percentage=0.5):

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-half-drop-panel"

        if pattern == "none":
            return (image, show_help, )

        # Convert to PIL image
        pil_img = tensor2pil(image)
        pil_img = pil_img.convert('RGBA')

        x, y = pil_img.size
        aspect_ratio = x / y
        d = int(drop_percentage * 100)

        panel_image = Image.new('RGBA', (x*2, y*2))

        if pattern == "half drop":
            panel_image.paste(pil_img, (0, 0))
            panel_image.paste(pil_img, (0, y))
            panel_image.paste(pil_img, (x, -y//2))
            panel_image.paste(pil_img, (x, y//2))
            panel_image.paste(pil_img, (x, 3*y//2))
        elif pattern == "quarter drop":
            panel_image.paste(pil_img, (0, 0))
            panel_image.paste(pil_img, (0, y))
            panel_image.paste(pil_img, (x, -3*y//4))
            panel_image.paste(pil_img, (x, y//4))
            panel_image.paste(pil_img, (x, 5*y//4))
        elif pattern == "custom drop %":
            panel_image.paste(pil_img, (0, 0))
            panel_image.paste(pil_img, (0, y))
            panel_image.paste(pil_img, (x, (d-100)*y//100))
            panel_image.paste(pil_img, (x, d*y//100))
            panel_image.paste(pil_img, (x, y + d*y//100))

        image_out = pil2tensor(panel_image.convert('RGB'))

        return (image_out, show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_DiamondPanel:

    @classmethod
    def INPUT_TYPES(s):

        patterns = ["none", "diamond"]

        return {"required": {
                    "image": ("IMAGE",),
                    "pattern": (patterns,),
                }
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "show_help", )
    FUNCTION = "make_panel"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")

    def make_panel(self, image, pattern, drop_percentage=0.5):

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Layout-Nodes#cr-diamond-panel"

        if pattern == "none":
            return (image, show_help, )

        # Convert to PIL image
        pil_img = tensor2pil(image)
        pil_img = pil_img.convert('RGBA')

        x, y = pil_img.size
        aspect_ratio = x / y
        d = int(drop_percentage * 100)

        panel_image = Image.new('RGBA', (x*2, y*2))

        if pattern == "diamond":

            diamond_size = min(x, y)
            diamond_width = min(x, y * aspect_ratio)
            diamond_height = min(y, x / aspect_ratio)

            diamond_mask = Image.new('L', (x, y), 0)
            draw = ImageDraw.Draw(diamond_mask)

            # Make sure the polygon points form a diamond shape
            draw.polygon([(x // 2, 0), (x, y // 2),
                          (x // 2, y), (0, y // 2)], fill=255)

            # Create a copy of the original image
            diamond_image = pil_img.copy()

            # Set alpha channel using the diamond-shaped mask
            diamond_image.putalpha(diamond_mask)

            # Paste the diamond-shaped image onto the panel_image at position (0, 0)
            panel_image.paste(diamond_image, (-x//2, (d-100)*y//100), diamond_image)
            panel_image.paste(diamond_image, (-x//2, d*y//100), diamond_image)
            panel_image.paste(diamond_image, (-x//2, y + d*y//100), diamond_image)
            panel_image.paste(diamond_image, (0, 0), diamond_image)
            panel_image.paste(diamond_image, (0, y), diamond_image)
            panel_image.paste(diamond_image, (x//2, (d-100)*y//100), diamond_image)
            panel_image.paste(diamond_image, (x//2, d*y//100), diamond_image)
            panel_image.paste(diamond_image, (x//2, y + d*y//100), diamond_image)
            panel_image.paste(diamond_image, (x, 0), diamond_image)
            panel_image.paste(diamond_image, (x, y), diamond_image)
            panel_image.paste(diamond_image, (3*x//2, (d-100)*y//100), diamond_image)
            panel_image.paste(diamond_image, (3*x//2, d*y//100), diamond_image)
            panel_image.paste(diamond_image, (3*x//2, y + d*y//100), diamond_image)

        image_out = pil2tensor(panel_image.convert('RGB'))

        return (image_out, show_help, )

#---------------------------------------------------------------------------------------------------------------------#
class CR_SelectISOSize:

    @classmethod
    def INPUT_TYPES(cls):

        sizes = list(iso_sizes.keys())

        return {
            "required": {
                "iso_size": (sizes, ),
            }
        }

    RETURN_TYPES =("INT", "INT","STRING", )
    RETURN_NAMES =("width", "height","show_help", )
    FUNCTION = "get_size"
    CATEGORY = icons.get("Comfyroll/Utils/Other")

    def get_size(self, iso_size):

        if iso_size in iso_sizes:
            width, height = iso_sizes[iso_size]
        else:
            print("Size not found.")

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Other-Nodes#cr-select-iso-size"

        return (width, height, show_help, )

#---------------------------------------------------------------------------------------------------------------------#
# based off ImageResizeKJv2 in https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/image_nodes.py
class CR_ImageResizeKJ:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "crop"], { "default": "resize" }),
                "pad_color": ("STRING", { "default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1 }),
            },
            "optional": {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "show_help", "width", "height", "mask")
    FUNCTION = "resize"
    CATEGORY = icons.get("Comfyroll/Graphics/Layout")
    DESCRIPTION = """
Resizes the image to the specified width and height.
Size can be retrieved from the input.

Keep proportions keeps the aspect ratio of the image, by
highest dimension.
"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, unique_id, device="cpu", mask=None):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        if width == 0:
            width = W
        if height == 0:
            height = H

        # Preserve original dimensions before any scaling/padding
        orig_w, orig_h = W, H

        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            if width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
            elif height == 0 and width != 0:
                ratio = width / W
                new_height = round(H * ratio)
            elif width != 0 and height != 0:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)

            if keep_proportion.startswith("pad"):
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        out_image = image.clone().to(device)

        if mask is not None:
            out_mask = mask.clone().to(device)
        else:
            out_mask = None

        # Track whether we're cropping
        pad_left_final = pad_right_final = pad_top_final = pad_bottom_final = 0
        final_w, final_h = width, height

        if keep_proportion == "crop":
            old_aspect = W / H
            new_aspect = width / height

            if old_aspect > new_aspect:
                crop_w = round(H * new_aspect)
                crop_h = H
            else:
                crop_w = W
                crop_h = round(W / new_aspect)

            if crop_position == "center":
                x = (W - crop_w) // 2
                y = (H - crop_h) // 2
            elif crop_position == "top":
                x = (W - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (W - crop_w) // 2
                y = H - crop_h
            elif crop_position == "left":
                x = 0
                y = (H - crop_h) // 2
            elif crop_position == "right":
                x = W - crop_w
                y = (H - crop_h) // 2

            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
            if mask is not None:
                out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)
            final_w, final_h = crop_w, crop_h
        else:
            pad_left_final = pad_right_final = pad_top_final = pad_bottom_final = 0
            if keep_proportion.startswith("pad"):
                pad_left_final = pad_left
                pad_right_final = pad_right
                pad_top_final = pad_top
                pad_bottom_final = pad_bottom
                final_w, final_h = width + pad_left + pad_right, height + pad_top + pad_bottom

        # Resize
        out_image = common_upscale(out_image.movedim(-1,1), width, height, upscale_method, crop="disabled").movedim(1,-1)

        if mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1,-1)[:, :, :, 0]
            else:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)

        # Apply padding
        if keep_proportion.startswith("pad"):
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height
                out_image, _ = self.pad(out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, "edge" if keep_proportion == "pad_edge" else "color")
                if mask is not None:
                    out_mask = out_mask.unsqueeze(1).repeat(1, 3, 1, 1).movedim(1,-1)
                    out_mask, _ = self.pad(out_mask, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, "edge" if keep_proportion == "pad_edge" else "color")
                    out_mask = out_mask[:, :, :, 0]
                else:
                    B, H_pad, W_pad, _ = out_image.shape
                    out_mask = torch.ones((B, H_pad, W_pad), dtype=out_image.dtype, device=out_image.device)
                    out_mask[:, pad_top:pad_top+height, pad_left:pad_left+width] = 0.0

        # --- BUILD SCENE GRAPH ---
        # Hash comes from first input image (source identity)
        input_hash = get_tensor_hash(image[0].unsqueeze(0))  # Use first frame's hash

        # Final canvas size
        canvas_w = out_image.shape[2]
        canvas_h = out_image.shape[1]

        # Position and size of content
        content_x = pad_left_final
        content_y = pad_top_final
        content_w = width
        content_h = height

        show_help_data = {
            "x": 0,
            "y": 0,
            "width": canvas_w,
            "height": canvas_h,
            "images": [
                {
                    "x": int(content_x),
                    "y": int(content_y),
                    "width": int(content_w),
                    "height": int(content_h),
                    "images": input_hash  # Scalar string hash
                }
            ]
        }

        show_help = json.dumps(show_help_data)

        # Memory reporting
        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except:
                pass

        return (
            out_image.cpu(),
            show_help,
            out_image.shape[2],
            out_image.shape[1],
            out_mask.cpu() if out_mask is not None else torch.zeros(64, 64, device=torch.device("cpu"), dtype=torch.float32)
        )


    def pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Parse background color
        bg_color = [int(x.strip())/255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "lanczos", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            def _gaussian_blur_nchw(img_nchw, sigma_px):
                if sigma_px <= 0:
                    return img_nchw
                radius = max(1, int(3.0 * float(sigma_px)))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
                k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)
                c = img_nchw.shape[1]
                kx = kx.repeat(c, 1, 1, 1)
                ky = ky.repeat(c, 1, 1, 1)
                img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
                img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
                return img_nchw

            out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = common_upscale(src_b, bg_w, bg_h, "bilinear", crop="disabled")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
                if pad_mode == "edge":
                    # Pad with edge color (mean)
                    top_edge = image[b, 0, :, :]
                    bottom_edge = image[b, H-1, :, :]
                    left_edge = image[b, :, 0, :]
                    right_edge = image[b, :, W-1, :]
                    out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                    out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                    out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                    out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                elif pad_mode == "edge_pixel":
                    # Pad with exact edge pixel values
                    for y in range(pad_top):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                    for y in range(pad_top+H, padded_height):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, H-1, :, :]
                    for x in range(pad_left):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                    for x in range(pad_left+W, padded_width):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                    out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                    out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                    out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                    out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                else:
                    # Pad with specified background color
                    out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)


#---------------------------------------------------------------------------------------------------------------------#
# based off LoadImagesFromFolderKJ in https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/image_nodes.py
# but with some additional changes
class CR_LoadImagesFromFolderKJ:
    # Dictionary to store folder hashes (for IS_CHANGED)
    folder_hashes = {}

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        if not os.path.isdir(folder):
            return float("NaN")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tga']
        include_subfolders = kwargs.get('include_subfolders', False)

        file_data = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        path = os.path.join(root, file)
                        try:
                            mtime = os.path.getmtime(path)
                            file_data.append((path, mtime))
                        except OSError:
                            pass
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    path = os.path.join(folder, file)
                    try:
                        mtime = os.path.getmtime(path)
                        file_data.append((path, mtime))
                    except OSError:
                        pass

        file_data.sort()

        combined_hash = hashlib.md5()
        combined_hash.update(folder.encode('utf-8'))
        combined_hash.update(str(len(file_data)).encode('utf-8'))

        for path, mtime in file_data:
            combined_hash.update(f"{path}:{mtime}".encode('utf-8'))

        current_hash = combined_hash.hexdigest()

        old_hash = cls.folder_hashes.get(folder)
        cls.folder_hashes[folder] = current_hash

        if old_hash == current_hash:
            return old_hash

        return current_hash

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 0, "min": -1, "step": 1}),
                "height": ("INT", {"default": 0, "min": -1, "step": 1}),
                "keep_aspect_ratio": (["crop", "pad", "stretch"],),
                "filter_select": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One filename, stem, or absolute path per line."
                }),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "fail_on_missing_filter": ("BOOLEAN", {"default": False}),
                "use_largest_size": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "count", "image_path", "show_help")
    OUTPUT_IS_LIST = (False, False, False, True, False)
    FUNCTION = "load_images"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """Loads images from a folder into a batch. Supports filtering via exact name/stem or absolute paths."""

    def load_images(self, folder, width, height, keep_aspect_ratio, filter_select,
                image_load_cap=0, start_index=0, include_subfolders=False, fail_on_missing_filter=False, use_largest_size=False):

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tga']

        # --- SCAN ONCE: Original logic preserved ---
        image_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(folder, file))

        dir_files = sorted(image_paths)
        # --- END SCAN ---

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # Parse filter lines

        selected_files = []
        warnings = []

        if not filter_select.strip():
            # No filter → use all scanned files
            selected_files = dir_files
        else:
            filter_lines = [line.strip() for line in filter_select.split('\n') if line.strip()]
            abs_path_regex = re.compile(r"^[/\\]|[a-zA-Z]:[/\\]")
            # Otherwise, apply filtering per line
            for line in filter_lines:
                is_absolute = abs_path_regex.match(line)
                resolved_path = line.replace('\\', '/')

                if is_absolute:
                    if not os.path.isfile(resolved_path):
                        raise FileNotFoundError(f"Absolute path image not found: {resolved_path}")
                    selected_files.append(resolved_path)
                else:
                    matched = False
                    for fp in dir_files:
                        fname = os.path.basename(fp)
                        stem = fname.split('.')[0]  # First part only
                        if fname == line or stem == line:
                            selected_files.append(fp)
                            matched = True
                            break
                    if not matched:
                        msg = f"[CR_LoadImagesFromFolderKJ] Filter '{line}' matched no files in '{folder}'."
                        print(msg)
                        warnings.append(msg)
                        if fail_on_missing_filter:
                            raise ValueError(msg)

        # Apply start_index and cap
        selected_files = selected_files[start_index:]
        if image_load_cap > 0:
            selected_files = selected_files[:image_load_cap]

        images = []
        masks = []
        image_path_list = []
        show_help_images_list = []

        pbar = ProgressBar(len(selected_files))

        if use_largest_size and (width <= 0 or height <= 0):
            # No explicit canvas size specified, attempt
            # to determine based off largest dimensions among selected images
            max_w = 0
            max_h = 0
            for image_path in selected_files:
                if os.path.isdir(image_path):
                    continue

                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)

                # target size routine
                target_w = width if width != -1 else i.size[0]
                target_h = height if height != -1 else i.size[1]
                if target_w == 0 and target_h > 0:
                    target_w = int(target_h * (i.size[0] / i.size[1]))
                elif target_w > 0 and target_h == 0:
                    target_h = int(target_w * (i.size[1] / i.size[0]))
                elif target_w == 0 and target_h == 0:
                    target_w, target_h = i.size

                if target_w > max_w:
                    max_w = target_w
                if target_h > max_h:
                    max_h = target_h

            width = max_w if width <= 0 else width
            height = max_h if height <= 0 else height


        target_w = width
        target_h = height

        for image_path in selected_files:
            if os.path.isdir(image_path):
                continue

            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)

            # target size routine
            target_w = width if width != -1 else i.size[0]
            target_h = height if height != -1 else i.size[1]
            if target_w == 0 and target_h > 0:
                target_w = int(target_h * (i.size[0] / i.size[1]))
            elif target_h == 0 and target_w > 0:
                target_h = int(target_w * (i.size[1] / i.size[0]))
            elif target_w == 0 and target_h == 0:
                target_w, target_h = i.size

            if i.size != (target_w, target_h):
                (i, cx, cy, cwidth, cheight) = self.resize_with_aspect_ratio(i, target_w, target_h, keep_aspect_ratio)

            # Convert to tensor
            image = i.convert("RGB")
            image_tensor = pil2tensor(image)
            images.append(image_tensor)

            # Create mask
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                if mask.shape != (target_h, target_w):
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
            else:
                mask = torch.zeros((target_h, target_w), dtype=torch.float32)

            masks.append(mask)
            image_path_list.append(image_path)

            # Record position for show_help
            img_hash = get_tensor_hash(image_tensor)
            show_help_images_list.append({
                "x": cx,
                "y": cy,
                "width": cwidth,
                "height": cheight,
                "images": img_hash
            })

            pbar.update(1)

        # Stack images
        if len(images) == 0:
            raise ValueError("No valid images to load.")

        show_help = json.dumps(
            {
                "x": 0,
                "y": 0,
                "width": target_w,
                "height": target_h,
                "images": show_help_images_list
            }
        )

        if len(images) == 1:
            result = (images[0], masks[0], 1, image_path_list, show_help)
        else:
            image_batch = torch.cat(images, dim=0)
            mask_batch = torch.stack(masks, dim=0)
            result = (image_batch, mask_batch, len(images), image_path_list, show_help)

        return result

    def resize_with_aspect_ratio(self, img, width, height, mode):
        if mode == "stretch":
            return img.resize((width, height), Image.Resampling.LANCZOS)

        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        target_ratio = width / height

        if mode == "crop":
            # Calculate dimensions for center crop
            if aspect_ratio > target_ratio:
                # Image is wider - crop width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                left = (new_width - width) // 2
                crop_data = (left, 0, left + width, height)
                return (img.crop(crop_data),
                        crop_data[0], crop_data[1],
                        crop_data[2]-crop_data[0], crop_data[3] - crop_data[1]
                        )
            else:
                # Image is taller - crop height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                top = (new_height - height) // 2
                crop_data = (0, top, width, top + height)
                return (img.crop(crop_data),
                        crop_data[0], crop_data[1],
                        crop_data[2]-crop_data[0], crop_data[3] - crop_data[1]
                        )

        elif mode == "pad":
            pad_color = self.get_edge_color(img)
            # Calculate dimensions for padding
            if aspect_ratio > target_ratio:
                # Image is wider - pad height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                padding = (height - new_height) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (0, padding))
                return (padded, 0, padding,  width, new_height)
            else:
                # Image is taller - pad width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                padding = (width - new_width) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (padding, 0))
                return (padded, padding, 0, new_width, height)
    def get_edge_color(self, img):
        from PIL import ImageStat
        """Sample edges and return dominant color"""
        width, height = img.size
        img = img.convert('RGBA')

        # Create 1-pixel high/wide images from edges
        top = img.crop((0, 0, width, 1))
        bottom = img.crop((0, height-1, width, height))
        left = img.crop((0, 0, 1, height))
        right = img.crop((width-1, 0, width, height))

        # Combine edges into single image
        edges = Image.new('RGBA', (width*2 + height*2, 1))
        edges.paste(top, (0, 0))
        edges.paste(bottom, (width, 0))
        edges.paste(left.resize((height, 1)), (width*2, 0))
        edges.paste(right.resize((height, 1)), (width*2 + height, 0))

        # Get median color
        stat = ImageStat.Stat(edges)
        median = tuple(map(int, stat.median))
        return median


#---------------------------------------------------------------------------------------------------------------------#
# MAPPINGS
#---------------------------------------------------------------------------------------------------------------------#
# For reference only, actual mappings are in __init__.py
'''
NODE_CLASS_MAPPINGS = {
    "CR Page Layout": CR_PageLayout,
    "CR Image Grid Panel": CR_ImageGridPanel,
    "CR Half Drop Panel": CR_HalfDropPanel,
    "CR Diamond Panel": CR_DiamondPanel,
    "CR Image Border": CR_ImageBorder,
    "CR Feathered Border": CR_FeatheredBorder,
    "CR Color Panel": CR_ColorPanel,
    "CR Simple Text Panel": CR_SimpleTextPanel,
    "CR Overlay Transparent Image": CR_OverlayTransparentImage,
    #"CR Simple Titles": CR_SimpleTitles,
    "CR Select ISO Size": CR_SelectISOSize,
}
'''

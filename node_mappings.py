try:
    from .nodes.nodes_core import *
    from .nodes.nodes_aspect_ratio import *
    from .nodes.nodes_list import *
    from .nodes.nodes_lora import *
    from .nodes.nodes_controlnet import *
    from .nodes.nodes_pipe import *
    from .nodes.nodes_sdxl import *
    from .nodes.nodes_model_merge import *
    from .nodes.nodes_upscale import *
    from .nodes.nodes_xygrid import *
    from .nodes.nodes_legacy import *
except ImportError:
    print("\033[34mComfyroll Studio: \033[92mFailed to load Essential nodes\033[0m")

try:
    from .nodes.nodes_graphics_matplot import *
    from .nodes.nodes_graphics_text import *
    from .nodes.nodes_graphics_layout import *
    from .nodes.nodes_graphics_filter import *
    from .nodes.nodes_graphics_template import *
    from .nodes.nodes_graphics_pattern import *
except ImportError:
    print("\033[34mComfyroll Studio: \033[92mFailed to load Graphics nodes\033[0m")

try:
    from .nodes.nodes_animation_interpolation import *
    from .nodes.nodes_animation_io import *
    from .nodes.nodes_animation_prompt import *
    from .nodes.nodes_animation_schedulers import *
    from .nodes.nodes_animation_schedules import *
    from .nodes.nodes_animation_lists import *
    from .nodes.nodes_animation_utils import *
    from .nodes.nodes_animation_cyclers import *
except ImportError:
    print("\033[34mComfyroll Studio: \033[92mFailed to load Animation nodes\033[0m")

try:
    from .nodes.nodes_utils_logic import *
    from .nodes.nodes_utils_index import *
    from .nodes.nodes_utils_conversion import *
    from .nodes.nodes_utils_random import *
    from .nodes.nodes_utils_text import *
    from .nodes.nodes_utils_other import *
except ImportError:
    print("\033[34mComfyroll Studio: \033[92mFailed to load Utility nodes\033[0m")

NODE_CLASS_MAPPINGS = {
    ### Core Nodes
    "CR Image Output": CR_ImageOutput,
    "CR Latent Batch Size": CR_LatentBatchSize,
    "CR Conditioning Mixer": CR_ConditioningMixer,
    "CR Select Model": CR_SelectModel,
    "CR Seed": CR_Seed,
    "CR Prompt Text": CR_PromptText,
    "CR Combine Prompt": CR_CombinePrompt,
    "CR VAE Decode": CR_VAEDecode,
    ### List Nodes
    "CR Text List": CR_TextList,
    "CR Prompt List": CR_PromptList,
    "CR Simple List": CR_SimpleList,
    "CR Float Range List": CR_FloatRangeList,
    "CR Integer Range List": CR_IntegerRangeList,
    "CR Load Text List": CR_LoadTextList,
    "CR Binary To Bit List": CR_BinaryToBitList,
    "CR Text Cycler": CR_TextCycler,
    "CR Value Cycler": CR_ValueCycler,
    ### List IO
    "CR Load Image List": CR_LoadImageList,
    "CR Load Image List Plus": CR_LoadImageListPlus,
    "CR Load GIF As List": CR_LoadGIFAsList,
    "CR Font File List": CR_FontFileList,
    ### List Utils
    "CR Batch Images From List": CR_MakeBatchFromImageList,
    "CR Intertwine Lists" : CR_IntertwineLists,
    "CR Repeater": CR_Repeater,
    "CR XY Product": CR_XYProduct,
    "CR Text List To String": CR_TextListToString,
    ### Aspect Ratio Nodes
    "CR SD1.5 Aspect Ratio": CR_AspectRatioSD15,
    "CR SDXL Aspect Ratio": CR_SDXLAspectRatio,
    "CR Aspect Ratio": CR_AspectRatio,
    "CR Aspect Ratio Banners": CR_AspectRatioBanners,
    "CR Aspect Ratio Social Media": CR_AspectRatioSocialMedia,
    "CR_Aspect Ratio For Print": CR_AspectRatioForPrint,
    ### Legacy Nodes
    "CR Image Size": CR_ImageSize,
    "CR Aspect Ratio SDXL": CR_AspectRatio_SDXL,
    "CR SDXL Prompt Mixer": CR_PromptMixer,
    "CR Seed to Int": CR_SeedToInt,
    ### ControlNet Nodes
    "CR Apply ControlNet": CR_ApplyControlNet,
    "CR Multi-ControlNet Stack": CR_ControlNetStack,
    "CR Apply Multi-ControlNet": CR_ApplyControlNetStack,
    ### LoRA Nodes
    "CR Load LoRA": CR_LoraLoader,
    "CR LoRA Stack": CR_LoRAStack,
    "CR Random LoRA Stack": CR_RandomLoRAStack,
    "CR Random Weight LoRA": CR_RandomWeightLoRA,
    "CR Apply LoRA Stack": CR_ApplyLoRAStack,
    ### Model Merge Nodes
    "CR Apply Model Merge": CR_ApplyModelMerge,
    "CR Model Merge Stack": CR_ModelMergeStack,
    ### Pipe Nodes
    "CR Data Bus In":CR_DataBusIn,
    "CR Data Bus Out":CR_DataBusOut,
    "CR 8 Channel In":CR_8ChannelIn,
    "CR 8 Channel Out":CR_8ChannelOut,
    "CR Module Pipe Loader": CR_ModulePipeLoader,
    "CR Module Input": CR_ModuleInput,
    "CR Module Output": CR_ModuleOutput,
    "CR Image Pipe In": CR_ImagePipeIn,
    "CR Image Pipe Edit": CR_ImagePipeEdit,
    "CR Image Pipe Out": CR_ImagePipeOut,
    "CR Pipe Switch": CR_InputSwitchPipe,
    ### SDXL Nodes
    "CR SDXL Prompt Mix Presets": CR_PromptMixPresets,
    "CR SDXL Style Text": CR_SDXLStyleText,
    "CR SDXL Base Prompt Encoder": CR_SDXLBasePromptEncoder,
    ### Upscale Nodes
    "CR Multi Upscale Stack": CR_MultiUpscaleStack,
    "CR Upscale Image": CR_UpscaleImage,
    "CR Apply Multi Upscale": CR_ApplyMultiUpscale,
    ### XY Grid Nodes
    "CR XY List": CR_XYList,
    "CR XY Interpolate": CR_XYInterpolate,
    "CR XY From Folder": CR_XYFromFolder,
    "CR XY Save Grid Image": CR_XYSaveGridImage,
    "CR XY Index": CR_XYIndex,
    #"CR XYZ Index": CR_XYZIndex,
    ### Graphics Pattern
    "CR Halftone Grid": CR_HalftoneGrid,
    "CR Color Bars": CR_ColorBars,
    "CR Style Bars": CR_StyleBars,
    "CR Checker Pattern": CR_CheckerPattern,
    "CR Polygons": CR_Polygons,
    "CR Color Gradient": CR_ColorGradient,
    "CR Radial Gradient": CR_RadialGradient,
    "CR Starburst Lines": CR_StarburstLines,
    "CR Starburst Colors": CR_StarburstColors,
    "CR Simple Binary Pattern": CR_BinaryPatternSimple,
    "CR Binary Pattern": CR_BinaryPattern,
    ### Graphics Shape
    "CR Draw Shape": CR_DrawShape,
    "CR Draw Pie": CR_DrawPie,
    "CR Random Shape Pattern": CR_RandomShapePattern,
    ### Graphics Text
    "CR Overlay Text": CR_OverlayText,
    "CR Draw Text": CR_DrawText,
    "CR Mask Text": CR_MaskText,
    "CR Composite Text": CR_CompositeText,
    #"CR Arabic Text RTL": CR_ArabicTextRTL,
    "CR Simple Text Watermark": CR_SimpleTextWatermark,
    "CR Select Font": CR_SelectFont,
    ### Graphics Filter
    "CR Halftone Filter": CR_HalftoneFilter,
    "CR Color Tint": CR_ColorTint,
    "CR Vignette Filter": CR_VignetteFilter,
    ### Graphics Layout
    "CR Flattened Layout Regions": CR_FlattenedLayoutRegionsJSON,
    "CR Image Hash": CR_GetImageHash,
    "CR Save Image Layout Hash": CR_SaveImageHashLayout,
    "CR Save Image Leaf Data Hash": CR_SaveImageLeafData,
    "CR Page Layout": CR_PageLayout,
    "CR Image Panel": CR_ImagePanel,
    "CR Image Grid Panel": CR_ImageGridPanel,
    "CR Image Border": CR_ImageBorder,
    "CR Feathered Border": CR_FeatheredBorder,
    "CR Simple Text Panel": CR_SimpleTextPanel,
    "CR Color Panel": CR_ColorPanel,
    "CR Overlay Transparent Image": CR_OverlayTransparentImage,
    "CR Overlay Expanded Transparent Image": CR_ExpandedOverlayTransparentImage,
    "CR Half Drop Panel": CR_HalfDropPanel,
    "CR Diamond Panel": CR_DiamondPanel,
    #"CR Simple Titles": CR_SimpleTitles,
    "CR KJ Image Resize": CR_ImageResizeKJ,
    ### Graphics Template
    "CR Simple Meme Template": CR_SimpleMemeTemplate,
    "CR Simple Banner": CR_SimpleBanner,
    "CR Comic Panel Templates": CR_ComicPanelTemplates,
    "CR Simple Image Compare": CR_SimpleImageCompare,
    "CR Thumbnail Preview": CR_ThumbnailPreview,
    "CR Seamless Checker": CR_SeamlessChecker,
    ### Utils Logic
    "CR Image Input Switch": CR_ImageInputSwitch,
    "CR Image Input Switch (4 way)": CR_ImageInputSwitch4way,
    "CR Latent Input Switch": CR_LatentInputSwitch,
    "CR Conditioning Input Switch": CR_ConditioningInputSwitch,
    "CR Clip Input Switch": CR_ClipInputSwitch,
    "CR Model Input Switch": CR_ModelInputSwitch,
    "CR ControlNet Input Switch": CR_ControlNetInputSwitch,
    "CR VAE Input Switch": CR_VAEInputSwitch,
    "CR Text Input Switch": CR_TextInputSwitch,
    "CR Text Input Switch (4 way)": CR_TextInputSwitch4way,
    "CR Switch Model and CLIP": CR_ModelAndCLIPInputSwitch,
    ### Utils Process
    "CR Batch Process Switch": CR_BatchProcessSwitch,
    "CR Img2Img Process Switch": CR_Img2ImgProcessSwitch,
    "CR Hires Fix Process Switch": CR_HiResFixProcessSwitch,
    ### Utils Index
    "CR Index": CR_Index,
    "CR Index Increment": CR_IncrementIndex,
    "CR Index Multiply": CR_MultiplyIndex,
    "CR Index Reset": CR_IndexReset,
    "CR Trigger": CR_Trigger,
    ### Utils Conversion
    "CR String To Number": CR_StringToNumber,
    "CR String To Combo": CR_StringToCombo,
    "CR Float To String": CR_FloatToString,
    "CR Float To Integer": CR_FloatToInteger,
    "CR Integer To String": CR_IntegerToString,
    "CR String To Boolean": CR_StringToBoolean,
    ### Utils Random
    "CR Random Hex Color": CR_RandomHexColor,
    "CR Random RGB": CR_RandomRGB,
    "CR Random Multiline Values": CR_RandomMultilineValues,
    "CR Random Multiline Colors": CR_RandomMultilineColors,
    "CR Random RGB Gradient": CR_RandomRGBGradient,
    "CR Random Panel Codes": CR_RandomPanelCodes,
    ### Utils Text
    "CR Text": CR_Text,
    "CR Multiline Text": CR_MultilineText,
    "CR Split String": CR_SplitString,
    "CR Text Concatenate": CR_TextConcatenate,
    "CR Text Replace": CR_TextReplace,
    "CR Text Length": CR_TextLength,
    "CR Text Operation": CR_TextOperation,
    "CR Text Blacklist": CR_TextBlacklist,
    "CR Save Text To File": CR_SaveTextToFile,
    ### Utils Conditional
    "CR Set Value On Boolean": CR_SetValueOnBoolean,
    "CR Set Value On Binary": CR_SetValueOnBinary,
    "CR Set Value on String": CR_SetValueOnString,
    "CR Set Switch From String": CR_SetSwitchFromString,
    ### Utils Other
    "CR Value": CR_Value,
    "CR Integer Multiple": CR_IntegerMultipleOf,
    "CR Clamp Value": CR_ClampValue,
    "CR Math Operation": CR_MathOperation,
    "CR Get Parameter From Prompt": CR_GetParameterFromPrompt,
    "CR Select Resize Method": CR_SelectResizeMethod,
    "CR Select ISO Size": CR_SelectISOSize,
    ### Animation Nodes
    # Schedules
    "CR Simple Schedule": CR_SimpleSchedule,
    "CR Central Schedule": CR_CentralSchedule,
    "CR Combine Schedules": CR_CombineSchedules,
    "CR Output Schedule To File": CR_OutputScheduleToFile,
    "CR Load Schedule From File": CR_LoadScheduleFromFile,
    "CR Schedule Input Switch": Comfyroll_ScheduleInputSwitch,
    "CR Bit Schedule": CR_BitSchedule,
    # Schedulers
    "CR Simple Value Scheduler": CR_SimpleValueScheduler,
    "CR Simple Text Scheduler": CR_SimpleTextScheduler,
    "CR Value Scheduler": CR_ValueScheduler,
    "CR Text Scheduler": CR_TextScheduler,
    "CR Load Scheduled Models": CR_LoadScheduledModels,
    "CR Load Scheduled LoRAs": CR_LoadScheduledLoRAs,
    "CR Prompt Scheduler": CR_PromptScheduler,
    "CR Simple Prompt Scheduler": CR_SimplePromptScheduler,
    # Prompt
    "CR Keyframe List": CR_KeyframeList,
    #"CR Load Prompt Style": CR_LoadPromptStyle,
    "CR Encode Scheduled Prompts": CR_EncodeScheduledPrompts,
    # Interpolation
    "CR Gradient Float": CR_GradientFloat,
    "CR Gradient Integer": CR_GradientInteger,
    "CR Increment Float": CR_IncrementFloat,
    "CR Increment Integer": CR_IncrementInteger,
    "CR Interpolate Latents": CR_InterpolateLatents,
    # Utils
    "CR Debatch Frames": CR_DebatchFrames,
    "CR Current Frame": CR_CurrentFrame,
    #"CR Input Text List": CR_InputTextList,
    # IO
    "CR Load Animation Frames": CR_LoadAnimationFrames,
    "CR Load Flow Frames": CR_LoadFlowFrames,
    "CR Output Flow Frames": CR_OutputFlowFrames,
    ### Legacy
    # Note: CR Prompt List and CR Text List names have been reused,
    # so the old versions of these nodes are no longer available
    "CR Prompt List Keyframes": CR_PromptListKeyframes,
    "CR Simple Prompt List": CR_SimplePromptList,
    "CR Simple Prompt List Keyframes": CR_SimplePromptListKeyframes,
    "CR Cycle Models": CR_CycleModels,
    "CR Cycle LoRAs": CR_CycleLoRAs,
    "CR Cycle Text": CR_CycleText,
    "CR Cycle Text Simple": CR_CycleTextSimple,
    "CR Cycle Images": CR_CycleImages,
    "CR Cycle Images Simple": CR_CycleImagesSimple,
    "CR Model List": CR_ModelList,
    "CR LoRA List": CR_LoRAList,
    "CR Text List Simple": CR_TextListSimple,
    "CR Image List": CR_ImageList,
    "CR Image List Simple": CR_ImageListSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ### Core Nodes
    "CR Image Output": "💾 CR Image Output",
    "CR Integer Multiple": "⚙️ CR Integer Multiple",
    "CR Latent Batch Size": "⚙️ CR Latent Batch Size",
    "CR Seed": "🌱 CR Seed",
    "CR Value": "⚙️ CR Value",
    "CR Conditioning Mixer": "⚙️ CR Conditioning Mixer",
    "CR Select Model": "🔮 CR Select Model",
    "CR Prompt Text": "⚙️ CR Prompt Text",
    "CR Combine Prompt": "⚙️ CR Combine Prompt",
    "CR VAE Decode": "⚙️ CR VAE Decode",
    ### List Nodes
    "CR Text List": "📜 CR Text List",
    "CR Prompt List": "📜 CR Prompt List",
    "CR Simple List": "📜 CR Simple List",
    "CR Float Range List": "📜 CR Float Range List",
    "CR Integer Range List": "📜 CR Integer Range List",
    "CR Load Value List": "📜 CR Load Value List",
    "CR Load Text List": "📜 CR Load Text List",
    "CR Binary To Bit List": "📜 CR Binary To Bit List",
    "CR Text Cycler": "📜 CR Text Cycler",
    "CR Value Cycler": "📜 CR Value Cycler",
    ### List IO
    "CR Load Image List": "⌨️ CR Load Image List",
    "CR Load Image List Plus": "⌨️ CR Load Image List Plus",
    "CR Load GIF As List": "⌨️ CR Load GIF As List",
    "CR Font File List": "⌨️ CR Font File List",
    ### List Utils
    "CR Batch Images From List": "🛠️ CR Batch Images From List",
    "CR Intertwine Lists" : "🛠️ CR Intertwine Lists",
    "CR Repeater": "🛠️ CR Repeater",
    "CR XY Product": "🛠️ CR XY Product",
    "CR Text List To String": "🛠️ CR Text List To String",
    ### Aspect Ratio Nodes
    "CR SD1.5 Aspect Ratio": "🔳 CR SD1.5 Aspect Ratio",
    "CR SDXL Aspect Ratio": "🔳 CR SDXL Aspect Ratio",
    "CR Aspect Ratio": "🔳 CR Aspect Ratio",
    "CR Aspect Ratio Banners": "🔳 CR Aspect Ratio Banners",
    "CR Aspect Ratio Social Media": "🔳 CR Aspect Ratio Social Media",
    "CR_Aspect Ratio For Print": "🔳 CR_Aspect Ratio For Print",
    ### Legacy Nodes
    "CR Image Size": "CR Image Size (Legacy)",
    "CR Aspect Ratio SDXL": "CR Aspect Ratio SDXL (Legacy)",
    "CR SDXL Prompt Mixer": "CR SDXL Prompt Mixer (Legacy)",
    "CR Seed to Int": "CR Seed to Int (Legacy)",
    ### ControlNet Nodes
    "CR Apply ControlNet": "🕹️ CR Apply ControlNet",
    "CR Multi-ControlNet Stack": "🕹️ CR Multi-ControlNet Stack",
    "CR Apply Multi-ControlNet": "🕹️ CR Apply Multi-ControlNet",
    ### LoRA Nodes
    "CR Load LoRA": "💊 CR Load LoRA",
    "CR LoRA Stack": "💊 CR LoRA Stack",
    "CR Random LoRA Stack": "💊 CR Random LoRA Stack",
    "CR Random Weight LoRA": "💊 CR Random Weight LoRA",
    "CR Apply LoRA Stack": "💊 CR Apply LoRA Stack",
    ### Model Merge Nodes
    "CR Apply Model Merge": "⛏️ CR Apply Model Merge",
    "CR Model Merge Stack": "⛏️ CR Model Merge Stack",
    ### Pipe Nodes
    "CR Data Bus In": "🚌 CR Data Bus In",
    "CR Data Bus Out": "🚌 CR Data Bus Out",
    "CR 8 Channel In": "🚌 CR 8 Channel In",
    "CR 8 Channel Out": "🚌 CR 8 Channel Out",
    "CR Module Pipe Loader": "✈️ CR Module Pipe Loader",
    "CR Module Input": "✈️ CR Module Input",
    "CR Module Output": "✈️ CR Module Output",
    "CR Image Pipe In": "🛩 CR Image Pipe In",
    "CR Image Pipe Edit": "🛩️ CR Image Pipe Edit",
    "CR Image Pipe Out": "🛩️ CR Image Pipe Out",
    "CR Pipe Switch": "🔀️ CR Pipe Switch",
    ### SDXL Nodes
    "CR SDXL Prompt Mix Presets": "🌟 CR SDXL Prompt Mix Presets",
    "CR SDXL Style Text": "🌟 CR SDXL Style Text",
    "CR SDXL Base Prompt Encoder": "🌟 CR SDXL Base Prompt Encoder",
    ### Upscale Nodes
    "CR Multi Upscale Stack": "🔍 CR Multi Upscale Stack",
    "CR Upscale Image": "🔍 CR Upscale Image",
    "CR Apply Multi Upscale": "🔍 CR Apply Multi Upscale",
    ### XY Grid Nodes
    "CR XY List": "📉 CR XY List",
    "CR XY Interpolate": "📉 CR XY Interpolate",
    "CR XY Index": "📉 CR XY Index",
    "CR XY From Folder": "📉 CR XY From Folder",
    "CR XY Save Grid Image": "📉 CR XY Save Grid Image",
    ### Graphics Pattern
    "CR Halftone Grid" : "🟫 CR Halftone Grid",
    "CR Color Bars" : "🟫 CR Color Bars",
    "CR Style Bars" : "🟪 CR Style Bars",
    "CR Checker Pattern": "🟦 CR Checker Pattern",
    "CR Polygons": "🟩 CR Polygons",
    "CR Color Gradient": "🟨 CR Color Gradient",
    "CR Radial Gradient": "🟨 CR Radial Gradient",
    "CR Starburst Lines": "🟧 CR Starburst Lines",
    "CR Starburst Colors": "🟧 CR Starburst Colors",
    "CR Simple Binary Pattern": "🟥 CR Simple Binary Pattern",
    "CR Binary Pattern": "🟥 CR Binary Pattern",
    ### Graphics Shape
    "CR Draw Shape": "🟡 CR Draw Shape",
    "CR Draw Pie": "🟢 CR Draw Pie",
    "CR Random Shape Pattern": "🔵 CR Random Shape Pattern",
    ### Graphics Text
    "CR Overlay Text": "🔤 CR Overlay Text",
    "CR Draw Text": "🔤️ CR Draw Text",
    "CR Mask Text": "🔤️ CR Mask Text",
    "CR Composite Text": "🔤️ CR Composite Text",
    "CR Simple Text Watermark": "🔤️ CR Simple Text Watermark",
    "CR Select Font":  "🔤️ CR Select Font",
    ### Graphics Filter
    "CR Halftone Filter": "🎨 Halftone Filter",
    "CR Color Tint": "🎨 CR Color Tint",
    "CR Vignette Filter": "🎨 CR Vignette Filter",
    ### Graphics Layout
    "CR Image Panel": "🌁 CR Image Panel",
    "CR Image Grid Panel": "🌁 CR Image Grid Panel",
    "CR Simple Text Panel": "🌁 CR Simple Text Panel",
    "CR Color Panel": "🌁 CR Color Panel",
    "CR Half Drop Panel": "🌁 CR Half Drop Panel",
    "CR Diamond Panel": "🌁 CR Diamond Panel",
    "CR Page Layout": "🌁 CR Page Layout",
    "CR Image Border": "🌁 CR Image Border",
    "CR Feathered Border": "🌁 CR Feathered Border",
    "CR Overlay Transparent Image": "🌁 CR Overlay Transparent Image",
    "CR KJ Image Resize": "🌁 CR KJ Image Resize",
    ### Graphics Template
    "CR Simple Meme Template": "📱 CR Simple Meme Template",
    "CR Simple Banner": "📱 CR Simple Banner",
    "CR Comic Panel Templates": "📱 CR Comic Panel Templates",
    "CR Simple Image Compare": "📱 CR Simple Image Compare",
    "CR Thumbnail Preview": "📱 CR Thumbnail Preview",
    "CR Seamless Checker": "📱 CR Seamless Checker",
    ### Utils Logic
    "CR Image Input Switch": "🔀 CR Image Input Switch",
    "CR Image Input Switch (4 way)": "🔀 CR Image Input Switch (4 way)",
    "CR Latent Input Switch": "🔀 CR Latent Input Switch",
    "CR Conditioning Input Switch": "🔀 CR Conditioning Input Switch",
    "CR Clip Input Switch": "🔀 CR Clip Input Switch",
    "CR Model Input Switch": "🔀 CR Model Input Switch",
    "CR ControlNet Input Switch": "🔀 CR ControlNet Input Switch",
    "CR VAE Input Switch": "🔀 CR VAE Input Switch",
    "CR Text Input Switch": "🔀 CR Text Input Switch",
    "CR Text Input Switch (4 way)": "🔀 CR Text Input Switch (4 way)",
    "CR Switch Model and CLIP": "🔀 CR Switch Model and CLIP",
    ### Utils Process
    "CR Batch Process Switch": "🔂 CR Batch Process Switch",
    "CR Img2Img Process Switch": "🔂 CR Img2Img Process Switch",
    "CR Hires Fix Process Switch": "🔂 CR Hires Fix Process Switch",
    ### Utils Index
    "CR Index":"🔢 CR Index",
    "CR Index Increment": "🔢 CR Index Increment",
    "CR Index Multiply": "🔢 CR Index Multiply",
    "CR Index Reset": "🔢 CR Index Reset",
    "CR Trigger": "🔢 CR Trigger",
    ### Utils Conversion
    "CR String To Number": "🔧 CR String To Number",
    "CR String To Combo": "🔧 CR String To Combo",
    "CR Float To String": "🔧 CR Float To String",
    "CR Float To Integer": "🔧 CR Float To Integer",
    "CR Integer To String": "🔧 CR Integer To String",
    "CR String To Boolean": "🔧 CR String To Boolean",
    ### Utils Random
    "CR Random Hex Color": "🎲 CR Random Hex Color",
    "CR Random RGB": "🎲 CR Random RGB",
    "CR Random Multiline Values": "🎲 CR Random Multiline Values",
    "CR Random Multiline Colors": "🎲 CR Random Multiline Colors",
    "CR Random RGB Gradient": "🎲 CR Random RGB Gradient",
    "CR Random Panel Codes": "🎲 CR Random Panel Codes",
    ### Utils Text
    "CR Text": "🔤 CR Text",
    "CR Multiline Text": "🔤 CR Multiline Text",
    "CR Split String": "🔤 CR Split String",
    "CR Text Concatenate": "🔤 CR Text Concatenate",
    "CR Text Replace": "🔤 CR Text Replace",
    "CR Text Blacklist": "🔤 Text Blacklist",
    "CR Text Length": "🔤 CR Text Length",
    "CR Text Operation": "🔤 CR Text Operation",
    "CR Save Text To File": "🔤 CR Save Text To File",
    ### Utils Conditional
    "CR Set Value On Boolean": "⚙️ CR Set Value On Boolean",
    "CR Set Value On Binary": "⚙️ CR Set Value On Binary",
    "CR Set Value on String": "⚙️ CR Set Value on String",
    "CR Set Switch From String": "⚙️ CR Set Switch From String",
    ### Utils Other
    "CR Integer Multiple": "⚙️ CR Integer Multiple",
    "CR Value": "⚙️ CR Value",
    "CR Clamp Value": "⚙️ CR Clamp Value",
    "CR Math Operation": "⚙️ CR Math Operation",
    "CR Get Parameter From Prompt": "⚙️ CR Get Parameter From Prompt",
    "CR Select Resize Method": "⚙️ CR Select Resize Method",
    "CR Select ISO Size": "⚙️ CR Select ISO Size",
    ### Animation Nodes
    # Schedules
    "CR Simple Schedule": "📋 CR Simple Schedule",
    "CR Central Schedule": "📋 CR Central Schedule",
    "CR Combine Schedules": "📋 CR Combine Schedules",
    "CR Output Schedule To File": "📋 CR Output Schedule To File",
    "CR Load Schedule From File": "📋 CR Load Schedule From File",
    "CR Schedule Input Switch": "📋 CR Schedule Input Switch",
    "CR Bit Schedule": "📋 CR Bit Schedule",
    # Schedulers
    "CR Simple Value Scheduler": "📑 CR Simple Value Scheduler",
    "CR Simple Text Scheduler": "📑 CR Simple Text Scheduler",
    "CR Value Scheduler": "📑 CR Value Scheduler",
    "CR Text Scheduler": "📑 CR Text Scheduler",
    "CR Load Scheduled Models": "📑 CR Load Scheduled Models",
    "CR Load Scheduled LoRAs": "📑 CR Load Scheduled LoRAs",
    "CR Prompt Scheduler": "📑 CR Prompt Scheduler",
    "CR Simple Prompt Scheduler": "📑 CR Simple Prompt Scheduler",
    # Prompt
    "CR Keyframe List": "📝 CR Keyframe List",
    #"CR Load Prompt Style": "📝 CR Load Prompt Style",
    "CR Encode Scheduled Prompts": "📝 CR Encode Scheduled Prompts",
    # Interpolation
    "CR Gradient Float": "🔢 CR Gradient Float",
    "CR Gradient Integer": "🔢 CR Gradient Integer",
    "CR Increment Float": "🔢 CR Increment Float",
    "CR Increment Integer": "🔢 CR Increment Integer",
    "CR Interpolate Latents": "🔢 CR Interpolate Latents",
    # Utils
    "CR Debatch Frames": "🛠️ CR Debatch Frames",
    "CR Current Frame": "🛠️ CR Current Frame",
    # IO
    "CR Load Animation Frames": "⌨️ CR Load Animation Frames",
    "CR Load Flow Frames": "⌨️ CR Load Flow Frames",
    "CR Output Flow Frames": "⌨️ CR Output Flow Frames",
    ### Legacy
    "CR Prompt List Keyframes": "CR Prompt List Keyframes (Legacy)",
    "CR Simple Prompt List": "CR Simple Prompt List (Legacy)",
    "CR Simple Prompt List Keyframes": "CR Simple Prompt List Keyframes (Legacy)",
    "CR Cycle Models": "CR Cycle Models (Legacy)",
    "CR Cycle LoRAs": "CR Cycle LoRAs (Legacy)",
    "CR Cycle Text": "CR Cycle Text (Legacy)",
    "CR Cycle Text Simple": "CR Cycle Text Simple (Legacy)",
    "CR Cycle Images": "CR Cycle Images (Legacy)",
    "CR Cycle Images Simple": "CR Cycle Images Simple (Legacy)",
    "CR Model List": "CR Model List (Legacy)",
    "CR LoRA List": "CR LoRA List (Legacy)",
    "CR Text List Simple": "CR Text List Simple (Legacy)",
    "CR Image List": "CR Image List (Legacy)",
    "CR Image List Simple": "CR Image List Simple (Legacy)",
    "CR Input Text List": "CR Input Text List (Legacy)",
}

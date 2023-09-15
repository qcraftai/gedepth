from functools import update_wrapper
from .densedepth_head import DenseDepthHead, DepthBaseDecodeHead
from .adabins_head import AdabinsHead
from .bts_head import BTSHead
from .dpt_head import DPTHead
from .binsformer_head import BinsFormerDecodeHead
from .maskpe_head import MaskedPE
# from .fcn_head import FCNHead
# from .ocr_head import OCRHead
from .cascade_decode_head import BaseCascadeDecodeHead
# from .decode_head_seg import BaseDecodeHead
# from .guidance_head import GuidanceHead
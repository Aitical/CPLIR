import torch
from torch.nn import functional as F
from collections import OrderedDict

from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

    @master_only
    def calculate_flops(self, input_dim=(3, 504, 504)):
        super().calculate_flops(input_dim=input_dim)

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt["network_g"].get("window_size", 16)
        temp_size = self.opt["val"].get("window_size", 16)
        window_size = max(window_size) if isinstance(
            window_size, list) else window_size
        window_size = max(temp_size, window_size)

        # re-padding image size with multi-scale window size
        if not isinstance(window_size, int):
            max_value = max(window_size)
            if not isinstance(max_value, int):
                max_value = max(max_value)
            if (max_value == 8) and (6 in window_size):
                window_size = 24
            else:
                window_size = max_value
        # window_size=16 #evaluation for Shift layer window size
        scale = self.opt.get("scale", 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[
            :, :, 0: h - mod_pad_h * scale, 0: w - mod_pad_w * scale
        ]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)

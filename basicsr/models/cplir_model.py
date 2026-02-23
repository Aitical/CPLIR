import torch
from torch.nn import functional as F
from collections import OrderedDict


from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class CPLIRModel(SRModel):

    @master_only
    def calculate_flops(self, input_dim=(3, 504, 504)):
        super().calculate_flops(input_dim=input_dim)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.neg_num = 4 # or 2
        self.ema_decay = train_opt.get('ema_decay', 0)

        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(
                self.opt['network_g']).to(self.device)

            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                param_key = self.opt['path'].get(
                    'param_key_g_ema', 'params_ema')
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get(
                    'strict_load_g', True), param_key)
            else:
                self.model_ema(0)  # copy net_g weight

            self.net_g_ema.eval()

            if self.opt.get('complie', False):
                self.net_g_ema = torch.compile(self.net_g_ema)

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(
                train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get('contrastive_opt'):
            self.cri_contrastive = build_loss(
                train_opt['contrastive_opt']).to(self.device)
        else:
            self.cri_contrastive = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter=None, tb_logger=None):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        neg_out_list = []
        with torch.no_grad():
            for _ in self.neg_num:
            neg_out_list.append(self.net_g(self.lq, is_neg=True))

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(
                self.output, neg_out_list)
            if l_percep is not None:
                l_total -= l_percep
                loss_dict['l_percep'] = l_percep

        l_total.backward()

        use_grad_clip = self.opt['train'].get('use_grad_clip', False)

        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(), use_grad_clip)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

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

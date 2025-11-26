from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import encoder
from decoder import LightDecoder
import os
import matplotlib.pyplot as plt
import numpy as np


class SparK(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='bn', sbn=False,
            use_targeted_masking=False, high_mask_ratio=0.8, low_mask_ratio=0.4,
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_h, self.fmap_w = input_size // downsample_raito, input_size // downsample_raito
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        
        self.use_targeted_masking = use_targeted_masking
        self.high_mask_ratio = high_mask_ratio
        self.low_mask_ratio = low_mask_ratio
        
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        
        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(self.hierarchy): # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            
            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            
            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()    # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)
            
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2
        
        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')
        
        # these are deprecated and would never be used; can be removed.
        self.register_buffer('imn_m', torch.empty(1, 3, 1, 1))
        self.register_buffer('imn_s', torch.empty(1, 3, 1, 1))
        self.register_buffer('norm_black', torch.zeros(1, 3, input_size, input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...
        
        self._printed_stats = False
        self._all_mask_ratios = []  
        self._theoretical_mask_ratios = []
    
    def mask(self, B: int, device, generator=None):
        """原始的随机遮盖方法"""
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
    
    def targeted_mask(self, B: int, device, masks_bhw, generator=None):
        """针对性遮盖方法：基于mask对重点区域和非重点区域使用不同的遮盖比例"""
        h, w = self.fmap_h, self.fmap_w
        
        # 将mask下采样到特征图大小
        masks_bhw = torch.nn.functional.interpolate(
            masks_bhw.unsqueeze(1).float(), 
            size=(h, w), 
            mode='nearest'
        ).squeeze(1)  # (B, h, w)
        
        # 计算每个样本的平均遮盖比例
        total_mask_ratios = []
        theoretical_mask_ratios = []
        
        active_masks = []
        for b in range(B):
            mask = masks_bhw[b]  # (h, w)
            
            # 计算重点区域和非重点区域
            high_priority_pixels = (mask > 0).nonzero(as_tuple=True)  # 重点区域像素坐标
            low_priority_pixels = (mask == 0).nonzero(as_tuple=True)  # 非重点区域像素坐标
            
            # 计算各区域的像素数量
            num_high_priority = len(high_priority_pixels[0])
            num_low_priority = len(low_priority_pixels[0])
            
            # 计算各区域需要保留的像素数量
            num_keep_high = max(0, int(num_high_priority * (1 - self.high_mask_ratio)))
            num_keep_low = max(0, int(num_low_priority * (1 - self.low_mask_ratio)))
            
            # 计算理论遮盖面积
            total_pixels = h * w
            theoretical_masked = num_high_priority * self.high_mask_ratio + num_low_priority * self.low_mask_ratio
            theoretical_mask_ratio = theoretical_masked / total_pixels
            theoretical_mask_ratios.append(theoretical_mask_ratio)
            
            # 创建遮盖mask
            active_mask = torch.zeros(h, w, dtype=torch.bool, device=device)
            
            # 对重点区域进行遮盖（排序法）
            if num_high_priority > 0 and num_keep_high > 0:
                perm = torch.randperm(num_high_priority, device=device)
                keep_idx = perm[:num_keep_high]
                high_keep_y = high_priority_pixels[0][keep_idx]
                high_keep_x = high_priority_pixels[1][keep_idx]
                active_mask[high_keep_y, high_keep_x] = True
            
            # 对非重点区域进行遮盖（排序法）
            if num_low_priority > 0 and num_keep_low > 0:
                perm = torch.randperm(num_low_priority, device=device)
                keep_idx = perm[:num_keep_low]
                low_keep_y = low_priority_pixels[0][keep_idx]
                low_keep_x = low_priority_pixels[1][keep_idx]
                active_mask[low_keep_y, low_keep_x] = True
            
            # 计算整体平均遮盖比例
            total_keep_pixels = active_mask.sum().item()
            total_mask_ratio = 1.0 - (total_keep_pixels / total_pixels)
            total_mask_ratios.append(total_mask_ratio)
            
            active_masks.append(active_mask)
        
        # 将结果堆叠成batch
        active_b1hw = torch.stack(active_masks).unsqueeze(1)  # (B, 1, h, w)
        
        # 计算平均遮盖比例
        avg_mask_ratio = sum(total_mask_ratios) / len(total_mask_ratios)
        avg_theoretical_mask_ratio = sum(theoretical_mask_ratios) / len(theoretical_mask_ratios)
        
        # 只在第一个batch时输出详细信息
        if not hasattr(self, '_printed_targeted_info'):
            print(f'[Targeted Masking] High ratio: {self.high_mask_ratio:.2f}, Low ratio: {self.low_mask_ratio:.2f}')
            print(f'[Targeted Masking] Expected avg ratio: {avg_theoretical_mask_ratio:.3f}, Actual avg ratio: {avg_mask_ratio:.3f}')
            self._printed_targeted_info = True
        
        self._all_mask_ratios.extend(total_mask_ratios)
        self._theoretical_mask_ratios.extend(theoretical_mask_ratios)
        return active_b1hw
    
    def visualize_masks(self, inp_bchw, masks_bhw, active_b1hw, save_dir="mask_visualizations", max_samples=20):
        os.makedirs(save_dir, exist_ok=True)
        B = inp_bchw.shape[0]
        C = inp_bchw.shape[1]
        H, W = inp_bchw.shape[2:]
        h, w = active_b1hw.shape[2:]
        p_h, p_w = H // h, W // w
        for i in range(min(B, max_samples)):
            img = inp_bchw[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            mask = masks_bhw[i].detach().cpu().numpy()
            active = active_b1hw[i, 0].detach().cpu().numpy()
            # 将 patch mask 上采样到原图大小
            active_up = np.kron(active, np.ones((p_h, p_w)))
            active_up = active_up[:H, :W]
            # 叠加显示
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(img)
            axs[0].set_title('Input')
            axs[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axs[1].set_title('Region Mask')
            axs[2].imshow(active_up, cmap='gray', vmin=0, vmax=1)
            axs[2].set_title('Patch Active')
            # 叠加显示
            overlay = img.copy()
            overlay[active_up < 0.5] = [1, 0, 0]  # 被遮盖patch涂红
            axs[3].imshow(overlay)
            axs[3].set_title('Overlay (Red=Masked)')
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'mask_vis_{i}.png'))
            plt.close(fig)
    
    def forward(self, inp_bchw: torch.Tensor, masks_bhw=None, active_b1ff=None, vis=False):
        # step1. Mask
        if active_b1ff is None:
            if self.use_targeted_masking and masks_bhw is not None:
                # 使用针对性遮盖策略
                active_b1ff: torch.BoolTensor = self.targeted_mask(inp_bchw.shape[0], inp_bchw.device, masks_bhw)
            else:
                # 使用原始随机遮盖
                active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)
        
        encoder._cur_active = active_b1ff    # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw
        
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        
        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff     # (B, 1, f, f)
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        
        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = self.patchify(inp_bchw), self.patchify(rec_bchw)   # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    # (B, L, C) ==mean==> (B, L)
        
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        
        if vis:
            masked_bchw = inp_bchw * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
            return inp_bchw, masked_bchw, rec_or_inp
        else:
            return recon_loss
    
    def patchify(self, bchw):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))  # (B, f*f, 3*downsample_raito**2)
        return bln
    
    def unpatchify(self, bln):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw
    
    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # 针对性遮盖策略
            'use_targeted_masking': self.use_targeted_masking,
            'high_mask_ratio': self.high_mask_ratio,
            'low_mask_ratio': self.low_mask_ratio,
            
            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys

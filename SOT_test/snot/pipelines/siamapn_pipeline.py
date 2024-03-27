from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pickle import NONE

import numpy as np

from snot.core.config_apn import cfg
from snot.models.siamapn_model import ModelBuilderAPN
from snot.trackers.siamapn_tracker import SiamAPNTracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class DNS_SiamAPNTracker(SiamAPNTracker):
    def __init__(self, model, enhancer=None, denoiser=None):
        super(DNS_SiamAPNTracker, self).__init__(model)
        
        self.model = model
        self.model.eval()

        self.enhancer = enhancer
        self.denoiser = denoiser

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        if self.enhancer is not None:
            z_crop = self.enhancer.enhance(z_crop)
        if self.denoiser is not None:
            z_crop = self.denoiser.denoise(z_crop)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        if self.enhancer is not None:
            x_crop = self.enhancer.enhance(x_crop)
        if self.denoiser is not None:
            x_crop = self.denoiser.denoise(x_crop)
        outputs = self.model.track(x_crop)

        self.anchors = self.generate_anchor() 
        score1 = self._convert_score(outputs['cls1'])*cfg.TRACK.w1
        score2 = self._convert_score(outputs['cls2'])*cfg.TRACK.w2
        score3=(outputs['cls3']).view(-1).cpu().detach().numpy()*cfg.TRACK.w3
        score=(score1+score2+score3)/3  

        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

 
class SiamAPNPipeline():
    def __init__(self, args, enhancer=None, denoiser=None):
        super(SiamAPNPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamAPN/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamAPN/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilderAPN()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.enhancer = enhancer
        self.denoiser = denoiser
        self.tracker = DNS_SiamAPNTracker(self.model, self.enhancer, self.denoiser)

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        self.tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        outputs = self.tracker.track(img)  
        pred_bbox = outputs['bbox']

        return pred_bbox



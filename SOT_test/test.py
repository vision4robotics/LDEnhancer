from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from snot.pipelines.pipeline_builder import build_pipeline
from snot.datasets import DatasetFactory, datapath
from denoiser.denoiser_builder import build_denoiser
from enhancer.enhancer_builder import build_enhancer
torch.set_num_threads(1) 

parser = argparse.ArgumentParser(description='Single object tracking')
parser.add_argument('--dataset', default='', type=str,
                    help='datasets')
parser.add_argument('--datasetpath', default='', type=str,
                    help='the path of datasets')
parser.add_argument('--config', default='./experiments/SiamAPN/config.yaml', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='./experiments/SiamAPN/model.pth', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--trackername', default='SiamAPN', type=str,
                    help='name of tracker')

parser.add_argument('--e_weights', default='./enhancer/SCT/ckpt/model.pth', type=str,
                    help='weights')
parser.add_argument('--enhancername', default='SCT', type=str,
                    help='name of enhancer')

parser.add_argument('--d_weights', default='./experiments/CDT/model.pth', type=str,
                    help='weights')
parser.add_argument('--denoisername', default='CDT', type=str,
                    help='name of denoiser')

parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()



def main():
    if args.enhancername.split('-')[0]:
        enhancer = build_enhancer(args)
    else:
        enhancer = None
    if args.denoisername.split('-')[0]:
        denoiser = build_denoiser(args)
    else:
        denoiser = None
    pipeline = build_pipeline(args, enhancer=enhancer, denoiser=denoiser)


    for dataset_name in args.dataset.split(','):
        # create dataset
        try:
            dataset_root = args.datasetpath + datapath[dataset_name]
        except:
            print('?')
        dataset = DatasetFactory.create_dataset(name=dataset_name,
                                            dataset_root=dataset_root,
                                            load_img=False)
        model_name = args.trackername
        
        # OPE tracking
        IDX = 0
        TOC = 0
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    pred_bbox = pipeline.init(img, gt_bbox)
                    pred_bboxes.append(pred_bbox)
                else:
                    pred_bbox = pipeline.track(img)
                    pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    try:
                        gt_bbox = list(map(int, gt_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    except:
                        pass
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results 
            model_path = os.path.join('results', dataset_name, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))
            IDX += idx
            TOC += toc
        print('Total Time: {:5.1f}s Average Speed: {:3.1f}fps'.format(TOC, IDX / TOC))
        fps_path = os.path.join('results', dataset_name, '{}.txt'.format(model_name))
        with open(fps_path, 'w') as f:
            f.write('Time:{:5.1f},Speed:{:3.1f}'.format(TOC, IDX / TOC))

if __name__ == '__main__':
    main()

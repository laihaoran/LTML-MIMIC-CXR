from .registry import DATASETS
from .xml_style import XMLDataset
import numpy as np
import mmcv
import pickle as pkl
import os.path as osp
import cv2

@DATASETS.register_module
class MIMICDataset(XMLDataset):
    with open('./appendix/mimic/class_name.pkl', 'rb') as f:
        CLASSES = pkl.load(f)

    CLASSES = tuple(CLASSES)

    def __init__(self, **kwargs):
        super(MIMICDataset, self).__init__(**kwargs)
        self.categories = self.CLASSES
    
    def load_annotations(self, ann_file, LT_ann_file=None):
            img_infos = []
            self.img_ids = mmcv.list_from_file(ann_file)

            for img_id in self.img_ids:
                filename = '{}.jpg'.format(img_id)
                img_path = osp.join(self.img_prefix,
                                    '{}'.format(filename))
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                width, height = img.shape
                width = int(width)
                height = int(height)
                img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
            return img_infos

    def get_ann_info(self, idx, epoch=0):
        img_id = self.img_infos[idx]['id']
        pkl_path = osp.join(self.img_prefix[:-7], 'mimic_pkl',
                            '{}.pkl'.format(img_id))
        with open(pkl_path, 'rb') as f:
            gt_labels = pkl.load(f)
        ann = dict(labels=gt_labels)
        return ann
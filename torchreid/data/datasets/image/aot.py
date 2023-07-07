from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

from ..dataset import ImageDataset


class AOTDataset(ImageDataset):
    # dataset_dir = 'aot_reid'

    def __init__(self, root='', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        # self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.dataset_dir = '/data2/workspace/datasets/re_id/aot_reid'
        train = []
        query = []
        gallery = []

        # train
        pid = 0
        train_dir = os.path.join(self.dataset_dir, 'train')
        for airborne in sorted(os.listdir(train_dir)):
            # pid = int(airborne.split('_')[-1])
            for img_name in os.listdir( os.path.join(train_dir, airborne) ):
                img_path = os.path.join(train_dir, airborne, img_name)
                train.append( (img_path, pid, 0) )
            pid += 1

        # query
        code_to_pid = dict()
        pid = 0
        query_dir = os.path.join(self.dataset_dir, 'query')
        for airborne in sorted(os.listdir(query_dir)):
            img_path = os.path.join(query_dir, airborne)
            code = int(airborne.split('_')[1])
            camid = int(airborne.split('_')[2][3:-4])
            query.append((img_path, pid, camid))
            code_to_pid[code] = pid
            pid += 1

        # gallery
        gallery_dir = os.path.join(self.dataset_dir, 'gallery')
        for airborne in sorted(os.listdir(gallery_dir)):
            code = int(airborne.split('_')[-1])
            for img_name in os.listdir( os.path.join(gallery_dir, airborne) ):
                img_path = os.path.join(gallery_dir, airborne, img_name)
                pid = code_to_pid[code]
                camid = int(img_name.split('_')[2][3:-4])
                gallery.append((img_path, pid, camid))

        super(AOTDataset, self).__init__(train, query, gallery, **kwargs)

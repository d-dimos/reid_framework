from __future__ import print_function, absolute_import

import logging

from .image import (
    GRID, PRID, CUHK01, CUHK02, CUHK03, MSMT17, CUHKSYSU, VIPeR, SenseReID,
    Market1501, DukeMTMCreID, University1652, iLIDS, AOTDataset
)
from .video import PRID2011, Mars, DukeMTMCVidReID, iLIDSVID
from .dataset import Dataset, ImageDataset, VideoDataset

from torchvision import transforms
from torchvision import datasets

from ..pixmix import PixMixDataset

__image_datasets = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'ilids': iLIDS,
    'sensereid': SenseReID,
    'prid': PRID,
    'cuhk02': CUHK02,
    'university1652': University1652,
    'cuhksysu': CUHKSYSU,
    'aot_reid': AOTDataset
}

__video_datasets = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    'dukemtmcvidreid': DukeMTMCVidReID
}


def init_image_dataset(name, args, apply_pixmix=False, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )

    X_dataset = __image_datasets[name](**kwargs)

    if args.mixing_set and apply_pixmix:
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
        mixing_set_transform = transforms.Compose(
            [transforms.Resize( (args.data.height+4, args.data.width+4) ),
             transforms.RandomCrop( (args.data.height, args.data.width) )])
        mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
        X_dataset = PixMixDataset(args, X_dataset, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
        logging.info(f'train_size: {len(X_dataset)}')
        logging.info(f'aug_size: {len(mixing_set)}')

    return X_dataset


def init_video_dataset(name, **kwargs):
    """Initializes a video dataset."""
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __video_datasets[name](**kwargs)


def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __image_datasets[name] = dataset


def register_video_dataset(name, dataset):
    """Registers a new video dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_video_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources=['new_dataset', 'ilidsvid']
        )
    """
    global __video_datasets
    curr_datasets = list(__video_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __video_datasets[name] = dataset

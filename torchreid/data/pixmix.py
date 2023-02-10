import numpy as np
import torch
from torchvision import transforms

from . import pixmix_utils as utils


class PixMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform PixMix."""

    def __init__(self, args, dataset, mixing_set, preprocess):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess
        self.args = args

    def __getitem__(self, i):

        if self.args.data.category == 'digits':
            x, y = self.dataset[i]
        elif self.args.data.category == 'person':
            x = self.dataset[i]['img']
            y = self.dataset[i]['pid']

        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        trans = transforms.ToPILImage()
        x = trans(x)
        return pixmix(self.args, x, mixing_pic, self.preprocess), y

    def __len__(self):
        return len(self.dataset)


def pixmix(args, orig, mixing_pic, preprocess):
    mixings = utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(args, orig))
    else:
        mixed = tensorize(orig)

    print("ONEONEONEON")
    print(mixed.shape)

    for _ in range(np.random.randint(args.mix_iters + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(args, orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

        print("TWOTWOTWO")
        print(aug_image_copy.shape)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)


def augment_input(args, image):
    aug_list = utils.augmentations_all if args.all_ops else utils.augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), args.aug_severity)

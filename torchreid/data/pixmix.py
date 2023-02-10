import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms

HEIGHT = 0
WIDTH = 0


class PixMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform PixMix."""

    def __init__(self, args, dataset, mixing_set, preprocess):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess
        self.args = args

        global HEIGHT, WIDTH
        HEIGHT = args.data.height
        WIDTH = args.data.width

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
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(args, orig))
    else:
        mixed = tensorize(orig)

    for _ in range(np.random.randint(args.mix_iters + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(args, orig))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)


def augment_input(args, image):
    aug_list = augmentations_all if args.all_ops else augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), args.aug_severity)


### UTILITIES ###
def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((WIDTH, HEIGHT),
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((WIDTH, HEIGHT),
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), WIDTH / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((WIDTH, HEIGHT),
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), HEIGHT / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((WIDTH, HEIGHT),
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


#########################################################
######################## MIXINGS ########################
#########################################################

def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1 ** a) * (img2.clip(1e-37) ** b)
    return out / 2


mixings = [add, multiply]


########################################
##### EXTRA MIXIMGS (EXPREIMENTAL) #####
########################################

def invert(img):
    return 1 - img


def screen(img1, img2, beta):
    img1, img2 = invert(img1), invert(img2)
    out = multiply(img1, img2, beta)
    return invert(out)


def overlay(img1, img2, beta):
    case1 = multiply(img1, img2, beta)
    case2 = screen(img1, img2, beta)
    if np.random.random() < 0.5:
        cond = img1 < 0.5
    else:
        cond = img1 > 0.5
    return torch.where(cond, case1, case2)


def darken_or_lighten(img1, img2, beta):
    if np.random.random() < 0.5:
        cond = img1 < img2
    else:
        cond = img1 > img2
    return torch.where(cond, img1, img2)


def swap_channel(img1, img2, beta):
    channel = np.random.randint(3)
    img1[channel] = img2[channel]
    return img1

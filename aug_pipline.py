import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance


def add_gaussian_noise(image, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_img = cv2.add(image, noise)
    return noisy_img

def adjust_brightness(image, factor=1.5):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    bright_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)

def random_crop(image, crop_size=(200, 200)):
    h, w = image.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError("Crop size exceeds image size")
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    return image[y:y+ch, x:x+cw]

def translate(image, tx=50, ty=30):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))

def rotate(image, angle=15):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def flip(image, mode='horizontal'):
    if mode == 'horizontal':
        return cv2.flip(image, 1)
    elif mode == 'vertical':
        return cv2.flip(image, 0)
    else:
        return image

def cutout(image, mask_size=50):
    h, w = image.shape[:2]
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)
    image_copy = image.copy()
    image_copy[y:y+mask_size, x:x+mask_size] = 0
    return image_copy

def augment_pipeline(image_path):
    image = cv2.imread(image_path)

    augmentations = {
        'gaussian_noise': add_gaussian_noise(image),
        'brightness': adjust_brightness(image),
        'cropped': random_crop(image, crop_size=(200, 200)),
        'translated': translate(image, tx=30, ty=20),
        'rotated': rotate(image, angle=30),
        'flipped_h': flip(image, mode='horizontal'),
        'flipped_v': flip(image, mode='vertical'),
        'cutout': cutout(image, mask_size=60)
    }

    for name, aug_img in augmentations.items():
        save_name = f"aug_{name}.jpg"
        cv2.imwrite(save_name, aug_img)
        print(f"Saved: {save_name}")


# example
if __name__ == "__main__":
    image_path = "test.jpg"
    augment_pipeline(image_path)
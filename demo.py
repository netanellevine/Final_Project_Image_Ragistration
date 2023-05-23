import os
import cv2
import numpy as np
from itertools import combinations
import imgaug.augmenters as iaa
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates



def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    """
    Add salt and pepper noise to image
    :param image: np.array of shape(height, width, channels)
    :param salt_prob: float, probability of salt noise
    :param pepper_prob: float, probability of pepper noise
    :return: np.array of shape(height, width, channels)
    """
    noisy_image = image.copy()
    total_pixels = image.size

    # Add salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords] = 1

    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords] = 0

    return noisy_image


def add_gaussian_noise(image, mean=0, std=1):
    """
    Add Gaussian noise to image
    :param image: np.array of shape(height, width, channels)
    :param mean: float, mean of the Gaussian distribution to generate noise
    :param std: float, standard deviation of the Gaussian distribution to generate noise
    :return: np.array of shape(height, width, channels)
    """
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian_noise
    return noisy_image


def rotate_image(image, angle):
    """
    Rotate the image by a certain angle
    :param image: np.array of shape(height, width, channels)
    :param angle: float, rotation angle in degrees
    :return: np.array of shape(height, width, channels)
    """
    # get the image size
    height, width = image.shape[:2]

    # compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def adjust_pixel_values(image, adjustment_value):
    """
    Adjust pixel values of the image
    :param image: np.array of shape(height, width, channels)
    :param adjustment_value: int, value to be added to the pixel values
    :return: np.array of shape(height, width, channels)
    """
    adjusted_image = image + adjustment_value
    return adjusted_image


def apply_cutout(image, size):
    """
    Apply cutout (random erasing) augmentation to the image
    :param image: np.array of shape(height, width, channels)
    :param size: int or tuple of int, size of the cutout
    :return: np.array of shape(height, width, channels)
    """
    cutout = iaa.Cutout(nb_iterations=(1, 3), size=size, squared=False)
    cutout_image = cutout.augment_image(image)
    return cutout_image


def apply_elastic_transform(image, alpha, sigma):
    """
    Apply elastic transformation to the image
    :param image: np.array of shape(height, width, channels)
    :param alpha: float, scale for the transformation
    :param sigma: float, standard deviation for the Gaussian filter
    :return: np.array of shape(height, width, channels)
    """
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


# specify your path
path = 'C:\\Users\\netan\\PycharmProjects\\dataManipulation\\data_before'

new_path = 'C:\\Users\\netan\\PycharmProjects\\dataManipulation\\data_after'

# get a list of all the image file names in the directory
image_files = os.listdir(path)

orig_images = []

# loop over each file
for image_file in image_files:
    # construct the full file path
    file_path = os.path.join(path, image_file)

    # read the image file
    image = cv2.imread(file_path)
    new_file_path = os.path.join(new_path, 'original_' + image_file)
    cv2.imwrite(new_file_path, image)
    orig_images.append(image)

    # apply the image manipulations
    # apply a Gaussian blur with a 7x7 kernel
    blurred_image = cv2.GaussianBlur(image, (13, 13), 0)
    new_file_path = os.path.join(new_path, f'blurred_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, blurred_image)

    # increase the contrast of the image by scaling pixel values by a factor of 1.5
    contrast_image = cv2.convertScaleAbs(blurred_image, alpha=1.5, beta=0)
    new_file_path = os.path.join(new_path, f'contrast_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, contrast_image)

    # increase the brightness of the image by adding 50 to pixel values
    bright_image = cv2.convertScaleAbs(contrast_image, alpha=1, beta=50)
    new_file_path = os.path.join(new_path, f'bright_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, bright_image)

    # rotate the image by a small angle
    rotated_image = rotate_image(bright_image, 5)  # 5 degrees, for example
    new_file_path = os.path.join(new_path, f'rotated_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, rotated_image)

    # add salt and pepper noise with a probability of 0.01 for both
    salt_pepper_image = add_salt_pepper_noise(rotated_image, 0.05, 0.05)
    new_file_path = os.path.join(new_path, f'salt_pepper_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, salt_pepper_image)

    # add Gaussian noise with a mean of 0 and a standard deviation of 0.1
    gaussian_image = add_gaussian_noise(salt_pepper_image, 0, 0.1)
    new_file_path = os.path.join(new_path, f'gaussian_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, gaussian_image)

    # apply pixel value adjustment
    adjusted_image = adjust_pixel_values(image, 50)  # increase pixel values by 50
    new_file_path = os.path.join(new_path, f'adjusted_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, adjusted_image)

    # apply cutout
    cutout_image = apply_cutout(adjusted_image, 0.2)  # apply cutout with size 20% of the original image size
    new_file_path = os.path.join(new_path, f'cutout_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, cutout_image)

    # apply elastic transformation
    transformed_image = apply_elastic_transform(cutout_image, 50, 3)  # apply elastic transformation with alpha=50 and sigma=3
    new_file_path = os.path.join(new_path, f'transformed_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, transformed_image)




def mixup(image1, image2, alpha):
    """
    Perform mixup augmentation on a pair of images
    :param image1: np.array of shape(height, width, channels)
    :param image2: np.array of shape(height, width, channels)
    :param alpha: float, mixup interpolation coefficient
    :return: np.array of shape(height, width, channels)
    """
    # Resize image2 to match the shape of image1
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    mixed_image = alpha * image1 + (1 - alpha) * image2_resized
    return mixed_image



# generate all possible pairs of images
image_pairs = list(combinations(orig_images, 2))

# perform mixup on all pairs of images
for i, (image1, image2) in enumerate(image_pairs):
    mixed_image = mixup(image1, image2, 0.5)  # 0.5 is the mixup interpolation coefficient, you can adjust it as needed

    # save the new image
    new_file_path = os.path.join(new_path, 'mixed_' + str(i) + '.jpg')
    cv2.imwrite(new_file_path, mixed_image)






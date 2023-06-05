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
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    salt_coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], salt_coords[2]] = 1

    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], pepper_coords[2]] = 0
    area = [1, 2, -1, -2]
    for i in area:
        try:
            noisy_image[pepper_coords[0]+i, pepper_coords[1]+i, pepper_coords[2]+i] = 0
            noisy_image[salt_coords[0]+i, salt_coords[1]+i, salt_coords[2]+i] = 1
        except:
            pass

    return noisy_image


def add_gaussian_noise(image, mean=0, std=10):
    """
    Add Gaussian noise to image
    :param image: np.array of shape(height, width, channels)
    :param mean: float, mean of the Gaussian distribution to generate noise
    :param std: float, standard deviation of the Gaussian distribution to generate noise
    :return: np.array of shape(height, width, channels)
    """
    gauss_noisy_image = image.copy()
    gaussian_noise = np.random.normal(mean, std, gauss_noisy_image.shape)
    gauss_noisy_image = gauss_noisy_image + gaussian_noise
    return gauss_noisy_image


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
    colors = [0, 50, 100, 150, 200, 255]  # Different grayscale intensities
    num_of_squares = 48
    num_of_iterations = num_of_squares // len(colors)
    cutout_image = image.copy()
    for i in range(len(colors)):
        color = colors[i]
        cutout = iaa.Cutout(nb_iterations=num_of_iterations, size=size, squared=True, fill_mode="constant", cval=color)
        cutout_image = cutout.augment_image(cutout_image)
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
path = 'Data_before_left/'

new_path = 'data_after_left/'

# get a list of all the image file names in the directory
image_files = os.listdir(path)

orig_images = []

# loop over each file
counter = 0

for image_file in image_files:
    # construct the full file path
    per_image_counter = 0
    file_path = os.path.join(path, image_file)

    # read the image file
    image = cv2.imread(file_path)
    image = cv2.resize(image,(256,256))
    new_file_path = os.path.join(new_path, 'original_' + image_file)
    success = cv2.imwrite(new_file_path, image)
    per_image_counter +=1
    orig_images.append(image)
    # apply the image manipulations
    # apply a Gaussian blur with a 7x7 kernel
    sizes = [5,7,9,11,13,15,17,19]
    for size in sizes:    
        blurred_image = cv2.GaussianBlur(image, (size, size), 0)
        new_file_path = os.path.join(new_path, f'blurred_image_{size}_{image_file}.jpg')
        success = cv2.imwrite(new_file_path, blurred_image)
        per_image_counter +=1

    # increase the contrast of the image by scaling pixel values by a factor of 1.5
    contrast_image = cv2.convertScaleAbs(image, alpha=2, beta=0)
    new_file_path = os.path.join(new_path, f'contrast_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, contrast_image)
    per_image_counter +=1

    # increase the brightness of the image by adding 80 to pixel values
    bright_image = cv2.convertScaleAbs(image, alpha=1, beta=80)
    new_file_path = os.path.join(new_path, f'bright_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, bright_image)
    per_image_counter +=1

    # rotate the image by a small angle
    angles = [3,6,-3,-6,9,-9]
    for angle in angles:
        rotated_image = rotate_image(image, angle)  # 6 degrees, for example
        new_file_path = os.path.join(new_path, f'rotated_image_{angle}_{image_file}.jpg')
        cv2.imwrite(new_file_path, rotated_image)
        per_image_counter +=1

    # add salt and pepper noise with a probability of 0.08 for both
    probabilities = [0.01,0.02,0.03,0.04,0.05]
    for prob_a in probabilities:
        for prob_b in probabilities:
            salt_pepper_image = add_salt_pepper_noise(image, prob_a, prob_b)
            new_file_path = os.path.join(new_path, f'salt_pepper_image_{prob_a}_{prob_b}_{image_file}.jpg')
            cv2.imwrite(new_file_path, salt_pepper_image)
            per_image_counter +=1

    # add Gaussian noise with a mean of 0 and a standard deviation of
    stds = [25,50,75,100]
    for std in stds:
        gaussian_image = add_gaussian_noise(image, 0, std)
        new_file_path = os.path.join(new_path, f'gaussian_image_{std}_{image_file}.jpg')
        cv2.imwrite(new_file_path, gaussian_image)
        per_image_counter +=1

    # # apply pixel value adjustment
    # adjusted_image = adjust_pixel_values(image, 70)  # increase pixel values by 50
    # new_file_path = os.path.join(new_path, f'adjusted_image_{image_file}.jpg')
    # cv2.imwrite(new_file_path, adjusted_image)

    # apply cutout
    cut = [0.03,0.05,0.07]
    for size in cut:
        cutout_image = apply_cutout(image, size)  # apply cutout with size 8% of the original image size
        new_file_path = os.path.join(new_path, f'cutout_image_{cut}_{image_file}.jpg')
        cv2.imwrite(new_file_path, cutout_image)
        per_image_counter +=1

    # apply elastic transformation
    transformed_image = apply_elastic_transform(image, 70, 6)  # apply elastic transformation with alpha=50 and sigma=3
    new_file_path = os.path.join(new_path, f'transformed_image_{image_file}.jpg')
    cv2.imwrite(new_file_path, transformed_image)
    per_image_counter +=1
    counter +=1
    print(f'finished {counter}/{len(image_files)} ~ {counter / len(image_files) * 100}%')
    print(f"created {per_image_counter} images!")




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



# # generate all possible pairs of images
# image_pairs = list(combinations(orig_images, 2))

# # perform mixup on all pairs of images
# for i, (image1, image2) in enumerate(image_pairs):
#     mixed_image = mixup(image1, image2, 0.5)  # 0.5 is the mixup interpolation coefficient, you can adjust it as needed

#     # save the new image
#     new_file_path = os.path.join(new_path, 'mixed_' + str(i) + '.jpg')
#     cv2.imwrite(new_file_path, mixed_image)






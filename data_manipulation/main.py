import os
import cv2
from PIL import ImageEnhance, Image
import numpy as np

PATH = 'C:\\Users\\netan\\PycharmProjects\\dataManipulation\\data_before'

image_files = os.listdir(PATH)

images = []

for image_file in image_files:
    curr_file_path = os.path.join(PATH, image_file)

    curr_image = cv2.imread(curr_file_path)

    images.append(curr_image)

print(len(images))
image = cv2.resize(images[0], (780, 900), cv2.INTER_CUBIC)
# cv2.imshow('image', image)
# cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (13, 13), 0)

def median_blur(img):
    return cv2.medianBlur(img, 13)

def avg_blur(img):
    return cv2.blur(img, (13, 13))


# gauss = gaussian_blur(image)
# cv2.imshow('gauss', gauss)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# median = median_blur(image)
# cv2.imshow('median', median)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# blur = avg_blur(image)
# cv2.imshow('blur', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def image_contrast(img):
    img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image (numpy array) to a PIL Image
    img_pil = Image.fromarray(img_cv)

    # Create an enhancer object
    enhancer = ImageEnhance.Contrast(img_pil)

    # Enhance the contrast of the image
    img_enhanced = enhancer.enhance(5.0)

    # If you want to save the enhanced image with OpenCV, you need to convert it back to a numpy array and BGR color order
    img_enhanced_cv = cv2.cvtColor(np.array(img_enhanced), cv2.COLOR_RGB2BGR)
    return img_enhanced_cv


# img_enhanced_cv = image_contrast(image)
# cv2.imshow('img_enhanced', img_enhanced_cv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def image_brightness(img):
    img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image (numpy array) to a PIL Image
    img_pil = Image.fromarray(img_cv)

    # Create an enhancer object
    enhancer = ImageEnhance.Brightness(img_pil)

    # Enhance the contrast of the image
    img_enhanced = enhancer.enhance(2.0)

    # If you want to save the enhanced image with OpenCV, you need to convert it back to a numpy array and BGR color order
    img_enhanced_cv = cv2.cvtColor(np.array(img_enhanced), cv2.COLOR_RGB2BGR)
    return img_enhanced_cv

# img_enhanced_cv = image_brightness(image)
# cv2.imshow('img_enhanced', img_enhanced_cv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
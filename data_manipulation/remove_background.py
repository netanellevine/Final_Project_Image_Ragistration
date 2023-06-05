import sys
import os
import cv2
import numpy as np
from PIL import Image
import torch
from u2net import U2NETP

# add the U-2-Net directory to the Python path
sys.path.append('/data_manipulation/U-2-Net/')


def remove_background(img):
    # load the U-2-Net model
    model_dir = '/data_manipulation/U-2-Net/saved_models/u2netp/U2NETP.pth'
    net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    # convert the image to a PyTorch tensor
    img_tensor = torch.from_numpy(np.array(img)).float()

    # pass the image through the model
    output = net(img_tensor)

    # process the output to create a binary mask
    mask = output.detach().numpy()
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    # apply the mask to the original image
    img_no_bg = img * mask

    return img_no_bg


# directory containing your images
image_dir = '/data_manipulation/data_before_back'

# loop over all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # read the image using OpenCV
        img = cv2.imread(os.path.join(image_dir, filename))

        # convert the image to PIL format for U-2-Net
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # use U-2-Net to remove the background
        img_no_bg = remove_background(img_pil)

        # create a new image with the same size as the original
        # fill it with your desired background color
        new_img = Image.new('RGB', img_no_bg.size, 'gray')

        # paste the original image onto the new image
        # this will only paste pixels that are not transparent
        new_img.paste(img_no_bg, (0, 0), img_no_bg)

        # save the new image
        new_img.save(os.path.join(image_dir, 'new_' + filename))

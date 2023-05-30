import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# add the U-2-Net directory to the Python path
sys.path.append('/data_manipulation/U-2-Net/')

# import the U-2-Net model
from model import U2NETP

# load the U-2-Net model
model_dir = '/data_manipulation/U-2-Net/saved_models/u2netp/u2netp.pth'
net = U2NETP(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

if torch.cuda.is_available():
    net.cuda()
net.eval()

# directory containing your images
image_dir = '/data_manipulation/data_before'

after_image_dir = '/data_manipulation/data_after'

# loop over all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.PNG'):
        print(f"Processing {filename}...")
        # read the image using PIL
        img = Image.open(os.path.join(image_dir, filename))

        # convert the image to RGB
        img = img.convert('RGB')

        # convert the image to a PyTorch tensor
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # pass the image through the model
        outputs = net(img_tensor)

        # select the first output tensor
        output = outputs[0]

        # process the output to create a binary mask
        mask = output.detach().cpu().numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # squeeze the mask to remove unnecessary dimensions
        mask = np.squeeze(mask)

        # resize the mask to the original image size
        mask = cv2.resize(mask, (img.width, img.height))

        # expand the mask to 3 channels
        mask = np.stack([mask, mask, mask], axis=-1)

        # apply the mask to the original image
        img_no_bg = img * mask

        # create a new image with the same size as the original
        # fill it with your desired background color
        new_img = Image.new('RGB', img.size, 'gray')

        # convert the image with no background to a PIL Image
        img_no_bg_pil = Image.fromarray((img_no_bg * 255).astype(np.uint8))

        # create a binary mask from the image with no background
        mask = img_no_bg_pil.convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')

        # paste the original image onto the new image
        # this will only paste pixels that are not transparent
        new_img.paste(img_no_bg_pil, (0, 0), mask)

        new_img = new_img.convert('RGB')

        # save the new image
        new_img.save(os.path.join(after_image_dir, 'new_' + filename))

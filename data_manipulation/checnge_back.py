from PIL import Image
import os

def replace_transparent_background_with_black(image_path):
    image = Image.open(image_path)

    # Check if the image has an alpha channel (transparency)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        has_transparency = True
    else:
        has_transparency = False

    # Create a new image with a black background and paste the original image onto it
    if has_transparency:
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))  # Black background
        background.paste(image, (0, 0), mask=image.convert("RGBA"))
        image = background

    # Save the modified image with a black background
    image.save(image_path)

# Set the directory path containing the images
directory_path = "/home/bravo8234/Desktop/study/final_project/git/Final_Project_Image_Ragistration/data_manipulation/data_before_left_no_bg"

# Iterate over the images in the directory and replace transparent background with black
for filename in os.listdir(directory_path):
    if filename.endswith(".png") or filename.endswith(".PNG"):
        image_path = os.path.join(directory_path, filename)
        replace_transparent_background_with_black(image_path)

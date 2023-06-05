import os
import shutil

def split_images(input_dir, output_dir, size):
  """Splits images in input_dir into subdirectories in output_dir, each with size images.

  Args:
    input_dir: The directory containing the images to split.
    output_dir: The directory to create the subdirectories in.
    size: The number of images to put in each subdirectory.
  """
  # Get the list of images in the input directory.
  images = os.listdir(input_dir)

  # Create the subdirectories in the output directory.
  for i in range(0, (len(images)//100)+1):
    subdir = os.path.join(output_dir, str(i))
    os.mkdir(subdir)

  # Move the images to the subdirectories.
  for image in images:
    index = images.index(image) // size
    dest = os.path.join(output_dir, str(index), image)
    shutil.copyfile(os.path.join(input_dir, image), dest)

if __name__ == "__main__":
  # Get the input and output directories from the user.
  input_dir = "/home/bravo8234/Desktop/study/final_project/git/Final_Project_Image_Ragistration/data_manipulation/data_after_back/"
  output_dir = "/home/bravo8234/Desktop/study/final_project/git/Final_Project_Image_Ragistration/split_back/"

  # Get the desired size of the subdirectories from the user.
  size = int(input("Enter the size of the subdirectories: "))

  # Split the images.
  split_images(input_dir, output_dir, size)

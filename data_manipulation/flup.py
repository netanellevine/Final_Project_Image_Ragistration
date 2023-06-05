import cv2
import os

def flip_images_in_directory(directory):
    # Get the list of files in the directory
    file_list = os.listdir(directory)
    print(f"files:{file_list}")
    for file_name in file_list:
        # Check if the file is an image
        if file_name.endswith(('.JPG', '.jpeg', '.png', '.bmp')):
            # Read the image
            image_path = os.path.join(directory, file_name)
            image = cv2.imread(image_path)
            
            # Flip the image horizontally
            flipped_image = cv2.flip(image, 1)
            
            # Save the flipped image with a new file name
            new_file_name = 'flipped_' + file_name
            flipped_image_path = os.path.join(directory, new_file_name)
            cv2.imwrite(flipped_image_path, flipped_image)
            print(f"Flipped image saved as: {flipped_image_path}")
            
    print("Flipping completed!")

# Provide the directory path where your images are stored
directory_path = '/home/bravo8234/Desktop/study/final_project/git/Final_Project_Image_Ragistration/data_manipulation/to_flip/'
flip_images_in_directory(directory_path)

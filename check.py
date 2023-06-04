import cv2

# Path to the image file
image_path = "/home/bravo8234/Desktop/study/final_project/git/Final_Project_Image_Ragistration/data_manipulation/data_before_back_no_bg/flipped_IMG_0694-removebg-preview.png"

# Read the image from the file
image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)

# Display the image
cv2.imshow("Image", image)

# Wait for the user to press a key
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
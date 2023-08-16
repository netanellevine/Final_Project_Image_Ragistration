# Registration of Medical Images for Breast Cancer: MRI and Mammography

![](https://user-images.githubusercontent.com/74298433/261010061-21222920-1d20-4dcc-a3cc-998d3bc3a8e2.png)
## Goal
Current breast cancer surgeries are often performed "blindly", which can result in the removal of excessive healthy breast tissue or missing the tumor margins. Our project addresses this challenge by providing surgeons with more precise parameters related to the tumor's geometry.

## What's New?
We introduce a novel dimension to the error calculation of our model. Our training focuses on the transformation we aim to achieve and the "target" images, which in our context are the final position (lying on the back).

## Introduction
- **Breast Cancer**: The leading cause of cancer among women globally. Despite breast conservative surgery combined with radiotherapy being the standard treatment, a significant number of patients end up with positive pathologic margins leading to high re-operation rates.
  
- **Surgeries**: There is a frequent removal of substantial amounts of healthy tissue during surgeries, thereby affecting the cosmetic outcome post-surgery. A method to precisely determine the location and geometry of cancerous lesions considering each patient's unique posture and image statistics can greatly enhance surgical outcomes.
  
- **Research**: This research focuses on MRI-Mammogram registration, forming part of a comprehensive framework. The potential impact of this project is massive, including fewer re-surgeries, reduced costs, shortened hospital stays, and lesser surgical complications.

## Goal (Detailed)
We are introducing a non-invasive technique for precise cancer analysis to enhance surgical outcomes. Our framework emphasizes adjusting the patient's posture for accurate tumor extraction. The end goal is to curtail re-operations, expenses, hospital stays, and surgical complications using an Augmented Reality tool.

## Methods
1. **Packages Used**: Initially, our research utilized VoxelMorph and PoseNet. However, we eventually transitioned to designing our neural network using the PyTorch library.

2. **Data Augmentation**: Given the scarcity of medical data, we curated our dataset and employed various augmentation techniques to expand it.

3. **Our Network**: Our custom neural network comprises four convolution-ReLU-max pooling blocks, followed by three linear-ReLU blocks. The final layer produces five values representing the affine transformation parameters. During training, we used both genuine ”back” images and ”left” images transformed into a lying position.

## Results
Our custom network, specifically designed based on our dataset, outperformed other approaches. We evaluated three kinds of outcomes:

1. A static transformation rotating the initial image by 90 degrees.
2. An image derived from the original image but transformed using the network's learned parameters.
3. The network's output image.

Our assessments using Mean Squared Error (MSE) and Normalized Cross-Correlation (NCC) on the test set indicated that the "net-grid sample" provided the best results in both metrics.

---

For additional details, visualizations, and updates, please refer to the accompanying documentation and our project's paper.

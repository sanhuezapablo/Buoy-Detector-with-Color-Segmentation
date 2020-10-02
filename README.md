# Buoy-Detection-using-Color-Segmentation

## **PROJECT DESCRIPTION**

The aim of this project is to perform Buoy Detection with the concept of color segmentation using Gaussian Mixture Models and Expectation Maximization techniques.

Since the video is shot underwater, conventional segmentation techniques involving color thresholding will not work well in such an environment, since noise and varying light intensities will render any hard-coded thresholds ineffective.

In such a scenario, we “learn” the color distributions of the buoys and use that learned model to segment them. We obtain a tight segmentation of each buoy for the entire video sequence by applying a tight contour (in the respective color of the buoy being segmented) around each buoy.

Please refer to [Project Report](https://github.com/adheeshc/Buoy-Detection-using-Color-Segmentation/blob/master/Report/Final%20Report.pdf) for further description

### Preparing the Input

<p align="center">
  <img src="/Images/Crop.png" alt="Input Prep">
</p>

- First, I have cropped out samples of the buoy and saved them for further training

#### Visualizing Histograms

<p align="center">
  <img src="/Images/avg_histo.png" alt="avg_histo">
</p>

- Then, for each buoy, I have computed and visualized the average color histogram for each channel of the sampled RGB images to get an idea of the output

#### 1D Gaussians

<p align="center">
  <img src="/Images/1D_gaussian.png" alt="1D Gaussian">
</p>

- Finally, I have designed and implemented a process to segment the buoys using a 1-D Gaussian to get a better idea of the output


<p align="center">
  <img src="/Images/output1.png" alt="bad output">
</p>

We see this method gives us bad results, however it does detect the buoys, we thus use this method as a stepping stone for our next step where we would train multiple gaussians onto the data.

### Expectation Maximization from Scratch

<p align="center">
  <img src="/Images/em.png" alt="EM">
</p>

- I have generated data samples from three 1-D Gaussians, with different means and variances.
- Next, the Expectation Maximization algorithm was Implemented
- Finally,the model parameters for the three Gaussians (i.e. the means and variances) are reecovered

### Learning Color Models

Now we extend the previous concepts and implementations of **1D gaussians** and **EM technique** to achieve the original goal of segmenting the buoys

#### 1D EM Implementation for Color Analysis

<p align="center">
  <img src="/Images/green_binary.png" alt="green_binary">
</p>

<p align="center">
  <img src="/Images/red_binary.png" alt="red_binary">
</p>

<p align="center">
  <img src="/Images/yellow_binary.png" alt="yellow_binary">
</p>

#### 3D EM Implementation Color Analysis

<p align="center">
  <img src="/Images/3d_gaussian.png" alt="Multivariate">
</p>

- For each color, I have already computed and visualized the color histogram for each channel of the sampled/cropped images
gathered during the data preparation phase. This will provide some intuition on the number of Gaussians [N] required to fit to the color histogram. I have also tried to determine the dimension [D] of each Gaussian for the model
- Then I use the previously implemented EM algorithm to compute the model parameters, i.e. the means and
variances of the N D-dimensional Gaussian.

### Final Buoy Detection

<p align="center">
  <img src="/Images/final.gif" alt="Final">
</p>


## **DEPENDANCIES**

- Python 3
- OpenCV
- Numpy
- Matplotlib
- Copy (built-in)


## **FILE DESCRIPTION**

- Code Folder/[Cropping.py](https://github.com/adheeshc/Buoy-Detection-using-Color-Segmentation/blob/master/Code/cropping.py) - This file is used for cropping out our original dataset
- Code Folder/[EM.py](https://github.com/adheeshc/Buoy-Detection-using-Color-Segmentation/blob/master/Code/EM.py) - This file is used for implementing the EM for a random set of datapoints
- Code Folder/[Project_3.py](https://github.com/adheeshc/Buoy-Detection-using-Color-Segmentation/blob/master/Code/Project_3.py) - This is the main file that implements the contours onto the buoys
- Code Folder/[(OPTIONAL) multivariate_gaussian.py](https://github.com/adheeshc/Buoy-Detection-using-Color-Segmentation/blob/master/Code/(OPTIONAL)%20multivariate_gaussian.py) - Here I implemented a 3D gaussian to fit the dataset

- Datasets folder - Contains 1 video input file, all the frames of the video and 3 folders containing images for Green, Yellow and Red Buoys 

- Images folder - Contains images for github use (can be ignored)

- Output folder - Contains output videos

- Report folder - Contains [Project Report](https://github.com/adheeshc/Buoy-Detection-using-Color-Segmentation/blob/master/Report/Final%20Report.pdf)

## **RUN INSTRUCTIONS**

- Make sure all dependancies are met
- Ensure the location of the input video files are correct in the code you're running
- Comment/Uncomment as reqd

- RUN Cropping.py if you want to create a new Dataset
- RUN EM.py if you want to implemenet EM from scratch for a randomly generated dataset
- RUN Project_3.py as is for Buoy Detection
- RUN (OPTIONAL) multivariate_gaussian.py if you want to test Buoy Detection it for a Multivariate Gaussian


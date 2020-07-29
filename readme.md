# Image Denoising Using Deep Convolutional Autoencoder with Feature Pyramids

## [[Paper]](http://journals.tubitak.gov.tr/elektrik/issues/elk-20-28-4/elk-28-4-20-1911-138.pdf)

## Citation
```
@article{cetinkaya2020denoising,
  title={Image denoising using deep convolutional autoencoder with feature pyramids},
  author={Cetinkaya, Ekrem and Kirac, M. Furkan},
  journal={Turkish Journal of Electrical Engineering \& Computer Sciences},
  volume={28},
  number={4},
  pages={2096-2109},
  year={2020},
  publisher={The Scientific and Technological Research Council of Turkey}
}
```

## Abstract
Image denoising is one of the fundamental problems in image processing field since it is required by many computer vision applications. Various approaches have been used in image denoising throughout the years from spatial filtering to model based approaches. Having outperformed all traditional methods, neural network based discriminative methods have gained popularity in the recent years. However, most of these methods still struggle to achieve flexibility against various noise levels and types. In this paper, we propose a deep convolutional autoencoder combined with a variant of feature pyramid network for image denoising. We use simulated data in Blender software along with corrupted natural images during training to improve robustness against various noise levels and types. Our experimental results show that proposed method can achieve competitive performance in blind Gaussian denoising with significantly less training time required compared to state-of-the-art methods. Extensive experiments showed us our method gives promising performance in wide range of noise levels with a single network.

## Network Architecture
The network takes a single-scale image as input with size *M x N* and passes it through four components: Early encoder, Feature Pyramid Network Component (FPNC), Smoothing layers and Decoder.

![Architecture](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/FPN_Architecture.png)

## Results
### Grayscale Image Denoising
![GrayDenoising](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/GrayDenoising/graydenoising.PNG)
#### PSNR Values in BSDS68
![GrayBSDSTable](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/GrayDenoising/graybsds.PNG)
#### PSNR Values in Set12
![GraySet12Table](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/GrayDenoising/grayset12.PNG)
***
### Color Image Denoising
![ColorDenoising](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/ColorDenoising/colordenoising.PNG)
#### PSNR Values
![ColorTable](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/ColorDenoising/colortable.PNG)
***
### Blender Render Denoising
![BlenderDenoising](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/BlenderDenoising/blenderdenoising.PNG)
### Blender Render Time Saving
![BlenderTime](https://github.com/ekremcet/AEFPNC/blob/master/RepoImages/BlenderTime/blendertime.PNG)

## Requirements
```
CUDA 10
Pytorch 1.0
Python 3.7
Pandas, Scikit-image, Numpy
```
## Prepare Data
First thing to do is to prepare the data and folders. **generate_noisy_data.py** script can be used for that: 
```console
foo@bar:~$ python generate_noisy_data.py
```
Default data folder for test images is */Datasets*. This should give noisy versions of test images along with their corresponding CSV files. It also prepares the folder structure of the project. 
```
AEFPNC
│ 
└───Checkpoints
└───Samples
│   └─── TestSamples
└───TestSets
│   └─── CBSDS68
│       └─── gaussian15
│       └─── gaussian25
│       | ...
│       | CBSDS68_gaussian15.csv
│       | CBSDS68_gaussian25.csv
│       | ...
│   └─── GCBSDS68
│       | ...
│   └─── Kodak24
│       | ...
│   └─── Set12
│       | ...
```
## Training Models
To train the network with your own dataset you need dataset of noisy and clean image pairs. The dataset we used for training our model can be downloaded from **[Here](https://drive.google.com/file/d/1_5ex8K54waX_19qcEJo_oVCweTMxmuMp/view?usp=sharing)**.

If you want to train the network with your own dataset, you can use **generate_noisy_data.py** script. For example:
```python
generate_noisy_set("./Datasets/BSDS500", "./TrainingSets", train=True, gray=False)
# Train argument will also generate Poisson noise
```
Above line is going to generate noisy dataset using images in **/Datasets/BSDS500** folder. You can then collect all the noisy images in a single folder that will be the noisy image folder for training. Then you should generate the **CSV** file for training dataset:
```python
generate_csv("./TrainingSet/Images", "./TrainingSet/GroundTruths", "TrainingSet")
# First argument is noisy image folder, second is ground truth folder, third is the csv name
```
Then you can run the training in **train.py** by changing the parameters of dataset loader in lines 20-21 to your corresponding dataset folders.


## Testing Models
Once you run the **generate_noisy_data.py**, you can run **evaluate.py** to generate denoised images from test sets given that trained model is in **/Checkpoints/AEFPNC.pth**. This should save denoised images in different noise levels into **/Samples/TestSamples/** folder along with corresponding PSNR and SSIM values for each image and noise level.

Pre-trained models for both color and grayscale denoising can be found in **Checkpoints** folder. It can also be downloaded from **[Here](https://drive.google.com/file/d/1SLwap1QrpMvnRWzHYhKsLpr72nD-37pM/view?usp=sharing)**. 
```python
evaluate_dataset("AEFPNC", "Kodak24")
#  First parameter is the name of checkpoint file and second parameter is the test dataset
```

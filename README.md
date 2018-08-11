# 3D Medical Segmentation GAN

|Arda Mavi|Deniz Yuret|
|:-:|:-:|
|Ayranci Anadolu High School|Koç University|

### 3D Liver Segmentation with GAN

#### This project created under `Koç University Summer Research Program`

## Architecture:
<img src="Assets/Model.png" width="600">

`Input Shape: n x 128 x 128 x 128`<br/>
`Output Shape: n x 128 x 128 x 128`

In this project we purpose to segmentation medical scans without unsuccessful loss functions in segmentation area like `Mean Squared Error` (not useful for segmentation) or `Dice Coefficient` (using for area comparison but not useful for gradient descent optimization function) and for benefit the best use of `GAN` algorithmic logic.

### ! Project and Documentations are under construction!

### New dataset expected...
For your dataset support: [Arda Mavi e-Mail](mailto:ardamavi2@gmail.com)
#### Currently Used Dataset: [DEU Liver Segmentation Dataset](https://eee.deu.edu.tr/moodle/mod/page/view.php?id=7872)

# Contents:
[For Users](#for-users)
- [Segmentation Scans](#segmentation-scans)

[For Developers](#for-developers)
- [Processing Dataset](#processing-dataset-command)
- [Model Training](#model-training)

! [Important Notes](#important-notes)

[To-Do List](#to-do-list)

# For Users

### Segmentation Scans:
`python3 predict.py <Scan_files_path>`


# For Developers

### Processing Dataset:
`python3 get_dataset.py`

### Model Training:
`python3 train.py`

# Important Notes
- Used Python 3.6.0 with Anaconda
- Install necessary modules with `sudo pip3 install -r requirements.txt` command.
- Load CUDNN(used `cudnn/7.0.5/cuda-9.0` for this project) module before training.


# To-Do List:
- [ ] Normalization of DICOM Liver datas.
- [ ] Optimize memory uses in test process.

# Lightweight UNet with Multi-module Synergy and Dual-domain Attention for Precise Skin Lesion Segmentation

### This is the official code repository for "Lightweight UNet with Multi-module Synergy and Dual-domain Attention for Precise Skin Lesion Segmentation".

## 1. Model structure.
- The overview of our model.
<img src="./img/tupianji.jpg">

## 2. Main Environments
- python 3.9
- [pytorch 2.0.1](https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-win_amd64.whl)
- [torchvision 0.15.2](https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl)

## 3. Prepare the dataset.

- The ISIC17 and ISIC18 datasets utilize the preprocessed datasets provided by [EGE-UNet](https://github.com/JCruan519/EGE-UNet) from [Google Drive](https://drive.google.com/file/d/1J6c2dDqX8qka1q4EtmTBA0w3Kez7-M6T/view?usp=sharing).

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png


## 4. Module decompression.

- Unpack the SwinT module in the modules file first.
- Update your file paths, including input and output paths in config_setting.py. This method trains the dataset once and requires removing the batch training method set in the train.py file.
- This work uses batch training with the option to modify the storage path in the train.py file.

## 5. Train the MSDA-UNet.
```
cd MSDA-UNet
```
```
python train.py
```

## 6. Obtain the outputs.
- After trianing, you could obtain the outputs in './results/'

## 7. Visualization.
<img src="./img/duibi.jpg">
>>>>>>> 7a6a448 (first commit)

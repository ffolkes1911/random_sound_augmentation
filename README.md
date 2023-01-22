# Description

Group university project where the effectiveness of mixup augmentation methods on the sound domain are analyzed.

# Usage 

`data_preprocessing.ipynb` - prepares required data files for provided datasets (ESC50 and US8k). Can mount Google drive to store data.  
`sound_classification.ipynb` - presents augmentation methods and performs training with ResNet18 model. Can grab data from Google drive.  
`Training_sound_classification.py` - performs training with ResNet18 model.  Created from `sound_classification.ipynb` to change recorded metrics, train/test split and to be able to run outside Jupyter.
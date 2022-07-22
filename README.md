# Deep-Learning-Radiomics-Research-in-Breast-Cancer
## Requirements
* python3.8.12
* pytorch1.10.2+cu113
* tensorboard 2.8.0
## Usage
### 1.dataset
CT images and clinicopathological information of the fourth and eleventh thoracic vertebrae in 421 patients with breast cancer.
### 2.train the model
The parameters of the model have been written into process.py.You need to train the model with the following commands：  
`$ python process.py`  
Determine whether the training data is the fourth level thoracic vertebra or the eleventh level thoracic vertebra by modifying the root variable in process.py. The save_path variable is the path of the trained weight file in process.py.



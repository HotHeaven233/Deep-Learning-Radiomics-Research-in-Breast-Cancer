# Deep-Learning-Radiomics-Research-in-Breast-Cancer
## Requirements
* python3.8.12
* pytorch1.10.2+cu113
* tensorboard 2.8.0
## Usage
### 1.dataset
CT images and clinicopathological information of the fourth and eleventh thoracic vertebrae in 421 patients with breast cancer.
### 2.train the densenet121 model
The parameters of the model have been written into process.py.You need to train the model with the following commands：  
`$ python process.py`  
Determine whether the training data is the fourth level thoracic vertebra or the eleventh level thoracic vertebra by modifying the root variable in process.py. The save_path variable is the path of the trained weight file in process.py.
### 3.run tensorboard
`$ tensorboard --logdir=./Result`
### 4.Extract image features
The img_root variable in predict.py determines the image to extract features (such as the test set of the fourth thoracic vertebra:'./data/T4-jipangji-predict',the test set of the eleven thoracic vertebra:'./data/T11-neizang-predict'), and the weights_path variable determines which trained network weight to use.  
`$ python predict.py`
### 5.RandomForest classification
The datasets models that combine the network features extracted at different levels with clinical information are run (in order: Clinical information combined DLR (T4&T11), Clinical information combined DLR (T4), Clinical information combined DLR (T11), Classification by clinical information).

  `cd model`
  `$ python rf_T4_T11.py`	
  `$ python rf_T4.py`	
  `$ python rf_T11.py`
  `$ python rf_clinical.py`

## ROC curve

> `cd model` 
> `$ python ROC.py`

## Result

| Methods                                   |      | AUC   | ACC  | SENS | SPEC | PPV  | NPV  |
| ----------------------------------------- | ---- | ----- | ---- | ---- | ---- | ---- | ---- |
| Clinical information combined DLR(T4&T11) | I-T  | 0.960 | 87.8 | 95.3 | 79.5 | 83.7 | 93.9 |
| Clinical information combined DLR(T4)     | I-T  | 0.940 | 85.4 | 88.4 | 82.1 | 84.4 | 86.5 |
| Clinical information combined DLR(T11)    | I-T  | 0.915 | 82.9 | 90.7 | 74.4 | 79.6 | 87.9 |
| Classification by clinical information    | I-T  | 0.874 | 79.3 | 86.8 | 72.7 | 73.3 | 86.5 |

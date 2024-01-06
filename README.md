# Image Classification of Stroke Blood Clot Origin

## Table of contents:

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Team]()

### Overview



### Project Structure

```
main
├─ Config
├─ data: dir containing 2 data samples
│  ├─  raw : original data
│  ├─  processed: Images Tiles
│  └─  processed-after-removing-low-score-imgs: Tiles after filtering 
├─ models: containing models checkpoints
├─ notebooks
│  ├─  Data Exploring.ipynb
│  ├─  Data Preprocessing Approach 1.ipynb
│  ├─  Data Preprocessing Approach 2.ipynb
│  ├─  EfficientNet_training_kaggle.ipynb
│  ├─  RestNet5_training_kaggle.ipynb
│  ├─  SqueezeNet_training_kaggle.ipynb
│  └─  test_for_all_models.ipynb
├─ reports
│  ├─  Paper.pdf
|  └─   figures
├─ src
│  ├─  preprocessing
│  │   ├─  preprocessing_approach_1.py
│  │   └─  preprocessing_approach_2.py
|  ├─  dataloaders
│  │   ├─  Dataloader.py
│  │   └─  Dataloader_with_aug.py
│  └─  models
│      ├─  EfficientNet.py
│      ├─  ResNet50.py
│      ├─  squeezeNet.py
│      └─  helpers
│          └─  FireModule.py
└─ README.md
```


### Team

First Semester - Artificial Neural Networks in Medicine (SBE4025) class project created by:

| Team Members' Names                                    | Code | 
| ------------------------------------------------------ | :-----: | 
| [Ahmed Hassan](https://github.com/ahmedhassan187) |    9202076    |
| [Habiba Fathallah](https://github.com/Habibafathalla)     |    9202458    |  
| [Rawan Mohamed](https://github.com/RawanFekry)   |    9202559    |  
| [Romaisaa Saad](https://github.com/Romaisaa)         |    9202564    |  
### Submitted to:

- Dr. Inas Yassine & Eng. Merna Biabers
  All rights reserved © 2024 to Team 3 - Systems & Biomedical Engineering, Cairo University (Class 2024)
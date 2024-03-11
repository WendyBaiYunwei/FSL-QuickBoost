* First unzip the `data` folder in this repository.
* Download the mini-imagenet dataset at https://lyy.mpi-inf.mpg.de/mtl/download/ and store it to the same `data` folder
* Edit `config.py` for the correct location of the `data` folder 
* Run `get_RF_data.py` to prepare the SimForest dataset. 
* You may run `test_rf.py` to train and test FSL-Forest as a standalone classifier
* Alternatively, you may directly test the ensemble. For 5-way-1-shot ensemble (mini-imagenet, relation network), please prepare a trained model via `miniimagenet_train_one_shot.py` to train, and use `test_ensemble.py` to test. 
* For 5-way-5-shot ensemble (mini-imagenet, relation network), please prepare a trained model via `miniimagenet_train_few_shot.py` to train, and use `test_ensemble_few_shot.py` to test.
* For more details on `mutual attention`, please refer to the `analysis` folder. 
  

* `miniimagenet_train_one_shot.py` and `miniimagenet_train_few_shot.py` are modified based on https://github.com/floodsung/LearningToCompare_FSL.

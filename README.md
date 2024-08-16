## FSL-QuickBoost

### Run code
* Download the mini-imagenet dataset at https://lyy.mpi-inf.mpg.de/mtl/download/ and store it to the same `data` folder
* Edit `config.py` for the correct location of the `data` folder 
* Run `get_rf_data.py` to prepare the QuickBoost dataset. 
* You may run `test_rf.py` to train and test QuickBoost as a standalone classifier
* Alternatively, you may directly test the ensemble. For 5-way-1-shot ensemble (mini-imagenet, relation network), please prepare a trained model via `miniimagenet_train_one_shot.py` to train, and use `test_ensemble.py` to test. 
* Note that the implemented FSL-Rectifier has a higher accuracy than reported results in paper due to addition of normalization to FSL-Forest features (See `rf_classifier.py`).

* `miniimagenet_train_one_shot.py` is modified based on https://github.com/floodsung/LearningToCompare_FSL.

  

### Optional reading
* QuickBoost's good results can be attributed to the nature of FSL classification, and encoder level feature diversity and classifier level diversity in our implementation. In our MM2024 paper, we present both feature and classifier ensemble as a whole package. As for classifier-level ensemble, FSL-Forest can be effective under suitable tuning. A quick experiment example is that, with FSL-Forest's number of estimators set to 200 and maximum features set to 20, QuickBoost can improve its cosine-similarity-based classifier of the same Res18 embedding from 47.90% to 49.66%. This improvement is caused by classifier level diversity; instead of making predictions based on cosine similarity that involves all feature channels, FSL-Forest makes prediction based on a subset of relatively important features, which is essentially different. We look forward to more in-depth future works exploring and analysing such potential. Meanwhile, we will try our best to responsibly maintain this code repository and address any issue raised.

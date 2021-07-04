# Bird-Song-Classification

<img align="center" alt="confmat" width=150% src="./wallpaper.png" />

We want to classify different birds species given their audio samples. We have extracted spectrograms of the audio samples and used them as features for classification. We can use the [British Birdsong Dataset](https://www.kaggle.com/rtatman/british-birdsong-dataset) available on Kaggle for this experiment.

Siamese Networks along with dilated 1D convolutions are used here to classify 9 different bird species.

<img align="center" alt="confmat" width="3000px" height="600" src="./confmat_test.png" />


Note: If you are having this error: AttributeError: module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'.
Type: 
pip uninstall tf-nightly <br>
pip uninstall tf-estimate-nightly
pip install tensorflow --upgrade --force-reinstall


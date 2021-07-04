# Bird-Song-Classification

<img align="center" alt="confmat" width=150% src="./wallpaper.png" />

### The goal is to classify different birds species based on their songs/calls. Spectrograms have been extracted from the audio samples and used as features for classification. A Siamese network based on 1D dilated convolutions is used here. Model is trained using triplet loss. I am using the [British Birdsong Dataset](https://www.kaggle.com/rtatman/british-birdsong-dataset) available on Kaggle for this experiment.

Download the data from [here](https://www.kaggle.com/rtatman/british-birdsong-dataset). This dataset is a subset of [Xeno-Canto database](https://www.xeno-canto.org/).
Siamese Networks along with dilated 1D convolutions are used here to classify 9 different bird species.

Confusion Matrix of testset: 
<img align="center" alt="confmat" width="3000px" height="600" src="./confmat_test.png" />

Scatter plot of embeddings after applying PCA: 
<img align="center" alt="scatter" height="600" src="./scatter.png" />


Note: If you are having this error: AttributeError: module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'. <br>
Type: 
pip uninstall tf-nightly <br>
pip uninstall tf-estimate-nightly <br>
pip install tensorflow --upgrade --force-reinstall <br>


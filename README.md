# Imperfect Gold Standard
## Environment
Python 3.7 and pytorch 1.9.1. More details are under requirements.txt, but some of them are redundant. //TODO
## How to run
1. Run `introduce_bias.py` to create csv files containing biased dataset
2. Run `train.py` with `IF_TUNE_HYPER_PARAM` set to True
3. Inspect the output result from it (require `WRITE_TEST_RESULT_TO_CSV` set to True)
4. Test the selected model checkpoint using `train.py` with `IF_TUNE_HYPER_PARAM` and `WRITE_TEST_RESULT_TO_CSV` set to True

Because previously I run them at my local machine so I do not write any sys.args. They can be run directly (`python introduce_bias.py`, `python train.py`).
### What does `IF_TUNE_HYPER_PARAM` do?
Setting it to true makes the script use validation data as test set, so it outputs acc on val set. Setting it to false makes the script use test set. 
## Dataset
Full dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia (too less val data to be used for tuning)

The samples I selected from the original dataset: https://drive.google.com/drive/folders/1axoxRGx0hE61erdbbsvX4Vs8zzkYppIK?usp=sharing (Unfortuanately I do not have the seed... so not replicable). After downloading from the google drive/having another train-val split from the original dataset, please put `all_images` dir under the root dir. 

`CXR_0_0_csv` contains the name of the images used in the experiment

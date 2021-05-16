# ATR

Based on the paper referenced from the [ATR paper](https://arxiv.org/pdf/2007.07690.pdf).

## Usage
Executing `python atr` will first run the training of densenet121 to create frozen weights to then be used to train atr, which runs second and outputs a frozen model upon completion.

When using the code, please make sure this script is in a directory with the dataset. Datasets are available online to download: [Densenet121](https://zenodo.org/record/3366686#.YKFoWKhKhhE) and [TW](https://zenodo.org/record/3923638#.YKFon6hKhhE). Make sure the all that data is placed in folders called ```fontgroupsdataset``` and ```twdataset``` respectfully. For both datasets place the CSVs provided in their dataset folder.

## JS
To run the ATR in JS, take the output .pb file and convert it using [tensorflowjs](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model).

## Tensorfair
For a live example of the ATR model visit [tensorfair](https://tensorfair.org/#atr).

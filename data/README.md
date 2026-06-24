# Dataset — State Farm Distracted Driver Detection

The dataset is **not included** in this repository (it is ~4 GB and licensed by
Kaggle). Download it yourself and place it here.

## 1. Get the data

Kaggle competition page:
https://www.kaggle.com/c/state-farm-distracted-driver-detection

Using the Kaggle CLI:

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle competitions download -c state-farm-distracted-driver-detection
unzip state-farm-distracted-driver-detection.zip -d data/
```

## 2. Expected folder layout

After unzipping, this folder must look like:

```
data/
├── README.md                 <- this file
├── imgs/
│   ├── train/
│   │   ├── c0/   (safe driving)
│   │   ├── c1/   (texting - right hand)
│   │   ├── c2/   (talking on phone - right hand)
│   │   ├── c3/   (texting - left hand)
│   │   ├── c4/   (talking on phone - left hand)
│   │   ├── c5/   (operating the radio)
│   │   ├── c6/   (drinking)
│   │   ├── c7/   (reaching behind)
│   │   ├── c8/   (hair and makeup)
│   │   └── c9/   (talking to passenger)
│   └── test/     (unlabeled images, optional)
└── driver_imgs_list.csv      (provided by Kaggle, optional)
```

The training scripts read `data/imgs/train` directly (configurable in
`config.yaml` under `paths.data_dir`).

## 3. Notes

- Each `c*` folder is one of the 10 behavior classes.
- The train/validation split is created automatically in code
  (`data.validation_split` in `config.yaml`); you do not need to split manually.
- The raw images are git-ignored, so they will never be committed.

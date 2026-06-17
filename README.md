# Fruit And Vegetable Classification With EfficientNetB0

This repository is aligned with the thesis topic:

- `TensorFlow / Keras`
- `EfficientNetB0` pretrained on ImageNet
- `224x224` RGB images
- `GlobalAveragePooling2D -> Dropout(0.55) -> Dense Softmax`
- `Adam(learning_rate=0.0005)`
- `L2 regularization = 0.001`
- `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
- output model saved as `.keras`

The target dataset for the thesis is the 8-class split inside:

- `Dataset_Fruit360_Image.zip`
- extracted root: `Dataset_New`
- classes: `apple`, `banana`, `cucumber`, `grape`, `mango`, `orange`, `pear`, `tomato`
- split folders: `train`, `validation`, `test`

## Folder structure

```text
dataset/
  train/
    apple/
    banana/
    ...
  validation/
    apple/
    banana/
    ...
  test/
    apple/
    banana/
    ...
```

## Extract The Dataset Zip

If your dataset is still a zip file, extract it first:

```bash
python extract_dataset_zip.py --zip-path C:/Users/DELL/Downloads/Dataset_Fruit360_Image.zip --output-dir data
```

The script will detect the folder that contains `train/validation/test`.

## Train

```bash
python train.py --dataset-root data/Dataset_New --output-dir artifacts
```

Main outputs:

- `artifacts/fruit8_efficientnet_group_regularized.keras`
- `artifacts/class_names.json`
- `artifacts/training_log.csv`
- `artifacts/training_curves.png`
- `artifacts/training_summary.json`

## Evaluate

```bash
python evaluate.py --dataset-root data/Dataset_New --model-path artifacts/fruit8_efficientnet_group_regularized.keras --class-names artifacts/class_names.json --output-dir outputs/evaluation
```

Main outputs:

- `outputs/evaluation/evaluation_summary.json`
- `outputs/evaluation/classification_report.json`
- `outputs/evaluation/classification_report.csv`
- `outputs/evaluation/confusion_matrix.csv`
- `outputs/evaluation/confusion_matrix.png`

## Predict One Image

```bash
python predict.py --image-path path/to/image.jpg --model-path artifacts/fruit8_efficientnet_group_regularized.keras --class-names artifacts/class_names.json
```

The script prints:

- predicted label
- confidence
- top 3 classes

## Streamlit Demo

```bash
streamlit run app.py
```

## Prepare Dataset Carefully

If you merge two datasets, do not split images randomly after merging. That can leak duplicates or near-duplicates across splits and give unrealistically high accuracy.

This repository now includes a duplicate-aware preparation step:

```bash
python prepare_dataset.py --config dataset_config.example.json --output-dir FruitDataset
```

What it does:

- merges multiple raw datasets into your shared target classes
- removes exact duplicate files using `md5`
- removes exact duplicate files that appear under conflicting labels
- groups images inside each class using `average hash`
- splits by group instead of by individual image
- writes `prepare_report.json` and `dataset_manifest.csv`

To audit an already prepared dataset:

```bash
python audit_dataset.py --dataset-dir FruitDataset --output-json outputs/dataset_audit_report.json
```

You want these numbers to be as close to zero as possible:

- `md5.overlap_group_count`
- `average_hash.overlap_group_count`

## Kaggle run

Enable Internet in Kaggle if you want to `git clone` directly from GitHub.
If you upload `Dataset_Fruit360_Image.zip` as a Kaggle dataset, extract it first.

```python
!git clone <YOUR_GIT_URL>
%cd EfficientNetB0
!python extract_dataset_zip.py --zip-path /kaggle/input/<dataset-name>/Dataset_Fruit360_Image.zip --output-dir /kaggle/working/data
!python audit_dataset.py --dataset-dir /kaggle/working/data/Dataset_New --output-json /kaggle/working/dataset_audit_report.json
!python train.py --dataset-root /kaggle/working/data/Dataset_New --output-dir /kaggle/working/artifacts
!python evaluate.py --dataset-root /kaggle/working/data/Dataset_New --model-path /kaggle/working/artifacts/fruit8_efficientnet_group_regularized.keras --class-names /kaggle/working/artifacts/class_names.json --output-dir /kaggle/working/evaluation
```

If Kaggle Internet is disabled and pretrained weights cannot be downloaded, use:

```python
!python train.py --dataset-root /kaggle/working/data/Dataset_New --output-dir /kaggle/working/artifacts --weights none
```

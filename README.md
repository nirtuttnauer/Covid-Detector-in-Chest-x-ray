# Chest X-Ray Classification Experiments

Exploratory deep-learning notebooks for chest X-ray classification and anomaly
visualization. The repository name refers to Covid-19, but the checked-in code
uses `NORMAL` / `PNEUMONIA` labels and, for multiclass experiments,
`NORMAL` / `BACTERIAL` / `VIRAL`. These labels do not establish a Covid-19 detector.
This is research/educational code, not a validated diagnostic tool.

## Repository contents

- `train.ipynb`: binary CNN, VGG16 transfer-learning and autoencoder experiments.
- `test.ipynb`: saved-model evaluation, prediction plots, embeddings, KNN and
  anomaly visualizations.
- `Deep Learning Covid Detector Report.docx`: accompanying project report.

Datasets, trained models and a dependency lockfile are not included.

## Local setup

```bash
git clone https://github.com/nirtuttnauer/Covid-Detector-in-Chest-x-ray.git
cd Covid-Detector-in-Chest-x-ray
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter tensorflow numpy matplotlib scikit-learn tqdm pillow h5py
python -m jupyter notebook
```

On Windows, activate with `.venv\Scripts\activate` instead.
The package list reflects notebook imports, not a tested or pinned environment.
The notebooks use legacy TensorFlow/Keras APIs (including
`tensorflow.keras.optimizers.legacy.Adam` and private `keras.src` imports), so
current package versions may require compatibility adjustments. Optional model
plotting also requires `pydot` and a Graphviz installation.

## Data and model preparation

Review cells before executing them; these are experiment notebooks, not a
turnkey pipeline. In particular:

1. Obtain a dataset with the class directories expected by the experiment.
   The binary paths in `train.ipynb` are:

   ```text
   dataset/chest_xray/
     train/
       NORMAL/
       PNEUMONIA/
     test/
       NORMAL/
       PNEUMONIA/
   ```

2. The extraction cell expects `dataset.zip` in the working directory. Run it
   only against a trusted archive and a disposable copy of the data. It moves
   images and recursively deletes directories. Some destinations use
   `val/NORMAL` for pneumonia images before deleting `val`; review and correct
   those paths before running. Multiclass cells additionally reference a nested
   `dataset/chest_xray/chest_xray/` tree and split pneumonia filenames containing
   `bacteria` or `virus` into separate directories.
3. Create output directories such as `models/1`, `models/2` and `models/4` before
   saving artifacts. Check each save/load filename: training and evaluation
   cells do not consistently use the same names or image dimensions.
4. `test.ipynb` contains Google Drive mounting and `/content/` paths. For local
   execution, skip the Drive-specific cell and replace its paths with your local
   dataset location. Supply the matching trained models before evaluation.
5. Run only the cells for the experiment you intend to reproduce. Training may
   be resource-intensive, and VGG16 with ImageNet weights may download weights.

## Evaluation limitations

Several experiments use the same test directory for validation during training
and final evaluation. Those scores are not an independent held-out estimate.
Multiclass prediction reports also need an unshuffled test generator so that
predictions align with `generator.classes`. Review these issues before drawing
conclusions from saved outputs; this README does not claim reproduced accuracy.

History loading uses NumPy's `allow_pickle=True`; load only trusted history
artifacts. No automated test suite or reproducible end-to-end environment is
currently included.

## Contributing and licensing

Focused fixes and reproducibility improvements are welcome via pull requests.
No license file is currently included; do not assume an MIT license grant.

# Data

This folder is where the dataset CSVs go. It is not versioned (see `.gitignore`) —
download it yourself:

1. Create an account with your `@epfl.ch` email at
   https://www.aicrowd.com/challenges/epfl-machine-learning-project-1
2. Download the training data from the challenge page.
3. Unzip it here so this folder contains exactly:
   - `x_train.csv`
   - `y_train.csv`
   - `x_test.csv`

Do not rename these files — `load_csv_data` in `helpers.py` expects these exact names.

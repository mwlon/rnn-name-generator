# RNN Name Generator

The purpose of this PyTorch repo is to train and validate a simple RNN model that predicts the next character in a sequence (or "name").
This model is then used for generating sample names.
The particular use case already built out is for startup names scraped from angel.co.

# Obtaining Startup Name Data

`cd startups`

`python scrape.py`

This will add lines to `raw_names.txt`.
Once you are satisfied the data you have scraped, sanitize and deduplicate the startup names with

`python clean_names.py`

This writes to `cleaned_names.txt`.
If angel.co changes their API, update the script accordingly.
If they really haven't made a substitute API, contact me for the dataset.

# Training and Validating

Create a train file and validation file, each of which has the same format as `cleaned_names.txt`: one line per name.
Update the hardcoded values in `train_val.py` to point to them.
Make any hyperparameter changes you want, such as changes to number of epochs, learning rate, or width/depth of the model.

`python train_and_val.py --train_path /path/to/train.txt ...`

This will both log and write training and validation metric results to `tracking.jsons`.

# Generating

Examples of generating names can be found in `generate_names.py`.

# Results

Example evaluation and generated startup names [here](http://graphallthethings.com/posts/startups).

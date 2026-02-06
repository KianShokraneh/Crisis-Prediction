# ”Listening In”: Social Signal Detection for Crisis Prediction
presentation link: https://www.canva.com/design/DAHAkQrw93s/dkOoMvHHXOMrgbz-Y0UfIg/edit?utm_content=DAHAkQrw93s&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
## Prerequisites

- Python 3.9+

## Quickstart (demo)

1. Create and activate a virtual environment.
2. Install dependencies.
   ```bash
   pip install -r Modules/OSOS/requirements.txt
   ```
3. Run the demo notebook.
   ```bash
   jupyter notebook demo/full_pipeline_demo.ipynb
   ```

## Run the OSOS pipeline (offline data)

The main pipeline expects a CSV named `Scrap_Results_bursty.csv` in your base data directory.
A sample file already exists in `data/`.

1. Create and activate a virtual environment.
2. Install dependencies.
   ```bash
   pip install -r Modules/OSOS/requirements.txt
   ```
3. Run the pipeline.
   ```bash
   python Modules/OSOS/Full_pipeline.py
   ```
4. When prompted, set the base path to `data`.
5. Provide the label text prompts for plots.
6. Provide an output filename for bursts, for example `bursts_output.csv`.

Outputs are written to the base path (`data/`), including burst plots and the burst periods CSV.

## Generate fresh input data

To rebuild the input data from public data:

1. Download and format Sentiment140.
   ```bash
   python Modules/OSOS/task1_download.py --base-path data --limit 1000
   ```
2. Create artificial bursty data (required by the pipeline).
   ```bash
   python Modules/OSOS/create_artificial_bursts.py --input data/Scrap_Results.csv --output data/Scrap_Results_bursty.csv --start-date 2009-04-01 --days 365 --burst-days 3 --burst-lengths 5,10,20 --burst-multiplier 4 --seed 42
   ```

## Evaluate sentiment models (optional)

1. Run model evaluation.
   ```bash
   python Modules/OSOS/task1_eval_models.py --base-path data --output-dir data
   ```
2. Render a Markdown table from results.
   ```bash
   python Modules/OSOS/task1_make_table.py
   ```

## Project structure and file definitions

- `Modules/` contains AI modules for the project.
- `Modules/OSOS/` is the active crisis signal detection pipeline.
- `Modules/OSOS/Full_pipeline.py` runs the end-to-end pipeline on offline data.
- `Modules/OSOS/task1_download.py` downloads and formats Sentiment140 into `Scrap_Results.csv`.
- `Modules/OSOS/create_artificial_bursts.py` generates `Scrap_Results_bursty.csv` used by the pipeline.
- `Modules/OSOS/task1_eval_models.py` evaluates sentiment models on Sentiment140.
- `Modules/OSOS/task1_make_table.py` converts evaluation results into `evaluation_table.md`.
- `Modules/OSOS/sentiment140_utils.py` shared helpers for the Sentiment140 dataset.
- `Modules/OSOS/requirements.txt` Python dependencies for OSOS.
- `data/` working dataset and outputs used by the pipeline.
- `demo/full_pipeline_demo.ipynb` demo notebook for the pipeline.
-

# Integrating Machine Learning and Cognitive Modeling of Decision Making
Implementation of various hybrid models based on MLBA and ML for human choice prediction.

## Reproducing the results of my thesis

### MLBA_HB
Run the follwoing R scripts to estimate MLBA using the hierarcical Bayesian method:
- `hb_mlba_code/E2-fit-predict-E1.R`
- `hb_mlba_code/E4-fit-predict-E3.R`

The results are already saved in `hb_mlba` folder.

### ML and Hybrid models
Run the following to train the models and get their results.
```bash
python run.py
```

After that, run the follwoing to generate the reports and the figures of the thesis saved in the `./out/res/` folder.
```bash
python paper_reports.py
```


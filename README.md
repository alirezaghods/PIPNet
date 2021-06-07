# PIP: Pictorial Interpretable Prototype Learning for Time Series Classification

The implimentation of PIP: Pictorial Interpretable Prototype Learning for Time Series Classification

## Data

We select three datasets: UCI-HAR, UCR-FordA, and UEA-SpokenArabicDigits for our expriment.

## Requirments

All python packages needed are listed in requirments.txt file and can be installed as follow:

```
conda config --append channels conda-forge
conda create --name <your env name> --file requirements.txt
```

## Expriments

To run each expriment you can following commands

```
python exp/uci-har-exp.py
python exp/ucr-fordA-exp.py
python exp/uea-arabic-exp.py
```

the result of all expriments save in 'trainin_history' folder.

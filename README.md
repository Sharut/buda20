Our codes are written in Python 3.7 for our winning participation at the PAN 2020 Profiling Fake News Spreaders on Twitter task

this repo has the following folders:

- 1_ngram_preprocessing: preprocessing scripts for the n-gram based models
- 2_stat_feature_engineering: feature extraction scripts for descriptive statistics based model
- 3_modeling: training scripts for the unique models
- 4_resampling: train and dev set construction for stacking model
- 5_stackingmodel: trining scripts for stacking model
- final software: the final script uploaded to TIRA
- models: the final trained models and vectorizers uploaded to TIRA for testing
- paper: paper describing our approach


# Data
training data available at https://zenodo.org/record/4039435#.X6LCj_NKi00


# Citation

If you use our code please cite our work.

    @InProceedings{lichouri:2020,
      author =              {Jakab Buda and Flora Bolonyai},
      booktitle =           {{CLEF 2020 Labs and Workshops, Notebook Papers}},
      crossref =            {pan:2020},
      editor =              {Linda Cappellato and Carsten Eickhoff and Nicola Ferro and Aur{\'e}lie N{\'e}v{\'e}ol},
      month =               sep,
      publisher =           {CEUR-WS.org},
      title =               {{An Ensemble Model Using N-grams and Statistical Features to Identify Fake News Spreaders on Twitter--Notebook for PAN at CLEF 2020}},
      url =                 {},
      year =                2020
      }


# Contribution

This code was developed by Flora Bolonyai and Jakab Buda

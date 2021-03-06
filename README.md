# H&M Personalized Fashion Recommendations

Kaggle [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview) 🥈 Silver Medal Solution 45/3006

![background](./imgs/img1.png)

![rank](./imgs/img2.png)

This repo contains our final solution. Big shout out to my wonderful teammates! [@zhouyuanzhe](https://github.com/zhouyuanzhe) [@tarickMorty](https://github.com/tarickMorty) [@Thomasyyj](https://github.com/Thomasyyj) [@ChenmienTan](https://github.com/ChenmienTan)

Our team ranked 45/3006 in the end with a LB score of 0.0292 and a PB score of 0.02996.

Our final solution contains 2 recall strategies and we trained 3 different ranking models (LGB ranker, LGB classifier, DNN) for each strategy.

Candidates from the two strategies are quite different so that ensembling the ranking results can help to improve the score. From our experiments, LB score of a single recall strategy can only reach 0.0286 and ensembling helps us to boost up to 0.0292. We also believe that ensembling can make our predicting result more robust.

Due to hardware limits (50G of RAM), we only generated avg 50 candidates for each user and used 4 weeks of data to train the models.

## Usage: 
1. Clone this repo
2. Create data folders in the structure shown below and copy the four .csv files from the original Kaggle competition dataset to `data/raw/`.
3. Pre-trained embeddings can be generated by [this notebook](https://github.com/Wp-Zhang/H-M-Fashion-RecSys/blob/main/notebooks/Embeddings.ipynb) or you can directly download them through the links below and put them in `data/external/`.
4. Run Jupyter Notebooks in `notebooks/`. Please note that features used by all models are generated in the `Feature Engineering` part in `LGB Recall 1.ipynb`, so make sure you run it first.

Google Drive Links of Pre-trained Embeddings
- [dssm_item_embd.npy](https://drive.google.com/file/d/13rGRbevjcd0yZdwuOTPmNyMOIx9WOLb9/view?usp=sharing)
- [dssm_user_embd.npy](https://drive.google.com/file/d/13nkDc7Dt6QtXx91i3sjnotQNGX2JpSk_/view?usp=sharing)
- [yt_item_embd.npy](https://drive.google.com/file/d/11Q8nWxOlSTspQwH9OGmR9vGoAqJ2wWbS/view?usp=sharing)
- [yt_user_embd.npy](https://drive.google.com/file/d/11OX9vuHmCrCk8Mcl6XA1TF0l0nBL___j/view?usp=sharing)
- [w2v_item_embd.npy](https://drive.google.com/file/d/1-8spKOVtb0jr5xYT8oMKMC5z3BPpCOU-/view?usp=sharing)
- [w2v_user_embd.npy](https://drive.google.com/file/d/1-6CAnA2_pHXrhCyplV-WsI9lreSf6Rm-/view?usp=sharing)
- [w2v_product_embd.npy](https://drive.google.com/file/d/1-R8Rww7QqHZOIcyIhZxEMiXRW1hzJ5wI/view?usp=sharing)
- [w2v_skipgram_item_embd.npy](https://drive.google.com/file/d/1-AmzbyCHx9i0CddZIdbqNJPAMXw3Kg34/view?usp=sharing)
- [w2v_skipgram_user_embd.npy](https://drive.google.com/file/d/1-8BpDQUn310Vns72t1up3uIOOnV_nR4h/view?usp=sharing)
- [w2v_skipgram_product_embd.npy](https://drive.google.com/file/d/1-QhHbFr16koCBL5OIMHxJX9ZAQJAhbHF/view?usp=sharing)
- [image_embd.npy](https://drive.google.com/file/d/1-WkIeInVvHJz4ScA3n-CRyVLQjW51gDH/view?usp=sharing)

Project Organization
------------

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── external       <- External data source, e.g. article/customer pre-trained embeddings.
    │   ├── interim        <- Intermediate data that has been transformed, e.g. Candidates generated form recall strategies.
    │   ├── processed      <- Processed data for training, e.g. dataframe that has been merged with generated features.
    │   └── raw            <- The original dataset.
    │
    ├── docs               <- Sphinx docstring documentation.
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to preprocess data
        │   ├── datahelper.py
        │   └── metrics.py
        │
        ├── features       <- Scripts of feature engineering
        │   └── base_features.py
        │
        └── retrieval      <- Scripts to generate candidate articles for ranking models
            ├── collector.py
            └── rules.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

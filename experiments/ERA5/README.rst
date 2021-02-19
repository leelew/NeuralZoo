ERA5 application for MetReg
===========================================================

`Lu Li <https://www.researchgate.net/profile/Lu_Li122>`_,
`Yongjiu Dai <https://www.researchgate.net/profile/Yongjiu_Dai2>`_,
`Wei Shangguan <https://www.researchgate.net/profile/Wei_Shangguan>`_,
`Jinjing Pan <https://www.researchgate.net/profile/Lu_Li122>`_

Pipelines
----------
1. Reading ERA5 dataset from nc format.

2. Preprocessing ERA5 dataset.(normalization, interplot, etc)

3. Generating training and validate dataset for ML/DL models.

4. Train and save models.

5. Inference and plot results.

Structure
----------
All experiments obey the **structure** shown as ERA5 application. It must contains the following files.

1. data_loader.py

    Read raw data from original folder and generate .npy (json or pickle) type first edition datasets.

2. data_generator.py

    Read un-processed datasets after exec `data_loader.py` and preprocessing (also splitting into sub-datasets for special experiments, such as world forecasting), finally, saving.

3. train.py

    Select models and train models, also save models.

4. inference.py

    Predict and benchmark the models.

5. plot.py

    Visualization.




NeuralZoo | A library for forecasting meteorological variables 
==============================================================
NOTE: This repo can NOT be used, please wait!  

`Lu Li <https://www.researchgate.net/profile/Lu_Li122>`_

 Read the `docs <https://github.com/leelew/NeuralZoo/blob/main/docs/pipeline.pdf>`_ | Try it by yourself!

Installation
-------------
MetReg support Python 3.6+. To install, you can use pip or conda. 

**Latest Release**

Install the latest release using pip.

.. code:: shell
   
   pip install NeuralZoo

**Development Version**

If you prefer the latest dev version, clone this repository and run the following command from the top-most folder of the repository. These commandwill build new environment and install **NeuralZoo**.

.. code:: shell
    
    make venv
    export PYTHONPATH=$PYTHONPATH:[abspath of home dir]/NeuralZoo

**Requirements**

**NeuralZoo** requires common used packages for machine learning. If you face any problems, try installing dependencies manually.

.. code:: shell
    
    make source
    make init

Citing
-------
If you find **NeuralZoo** useful for your research work, please cite us as follows:

* **HybridHydro**: Li, Lu et al.(2022) "Soil Moisture Forecasting integrating Physical-based Model with Deep Learning." Journal of Hydrometeorology.

* **CLSTM**: Li, Lu et al.(2022). "Causality-Structured Deep Learning for Soil Moisture Predictions." Journal of Hydrometeorology.

* **AttConvLSTM**: Li, Lu et al.(2022). "Multistep forecasting of soil moisture using a spatiotemporal deep encoder-decoder networks." Journal of Hydrometeorology.

* **RF-Granger**: Li, Lu, et al.(2020). "A causal inference model based on random forests to identify the effect of soil Moisture on precipitation." Journal of Hydrometeorology.

* **Comparative study**: Pan, J., et al.(2019). Using data‐driven methods to explore the predictability of surface soil moisture with FLUXNET site data. Hydrological Processes.






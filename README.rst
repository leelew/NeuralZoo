
MetReg | A library for forecasting meteorological variables 
===========================================================

`Lu Li <https://www.researchgate.net/profile/Lu_Li122>`_,
`Yongjiu Dai <https://www.researchgate.net/profile/Yongjiu_Dai2>`_,
`Wei Shangguan <https://www.researchgate.net/profile/Wei_Shangguan>`_,
`Jinjing Pan <https://www.researchgate.net/profile/Lu_Li122>`_

 Read the `docs <https://github.com/leelew/MetReg/blob/main/docs/pipeline.pdf>`_ | Try it by yourself!

Installation
-------------
MetReg support Python 3.6+. To install, you can use pip or conda. 

**Latest Release**

Install the latest release using pip.

.. code:: shell
   
   pip install MetReg

**Development Version**

If you prefer the latest dev version, clone this repository and run the following command from the top-most folder of the repository. These commandwill build new environment and install **MetReg**.

.. code:: shell
    
    make venv
    export PYTHONPATH=$PYTHONPATH:[abspath of home dir]/MetReg

**Requirements**

**MetReg** requires common used packages for machine learning. If you face any problems, try installing dependencies manually.

.. code:: shell
    
    make source
    make init

Citing
-------
If you find **MetReg** useful for your research work, please cite us as follows:

* **Comparative study**: Li, Lu et al.(2021) "Evaluation of machine learning methods for extended-range forecasting of soil moisture."

* **AttConvLSTM**: Li, Lu et al.(2021). "Multistep forecasting of soil moisture using a spatiotemporal deep encoder-decoder networks." Journal of Hydrometeorology.

* **RF-Granger**: Li, Lu, et al.(2020). "A causal inference model based on random forests to identify the effect of soil Moisture on precipitation." Journal of Hydrometeorology 21.5: 1115-1131.

* **Comparative study**: Pan, J., et al.(2019). Using data‐driven methods to explore the predictability of surface soil moisture with FLUXNET site data. Hydrological Processes, 33(23), 2978-2996.






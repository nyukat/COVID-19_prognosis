# An artificial intelligence system for predicting the deterioration of COVID-19 patients in the emergency department

[![DOI](https://zenodo.org/badge/309754870.svg)](https://zenodo.org/badge/latestdoi/309754870)

## Introduction
This code is an implementation of the AI system as described in [our paper](https://arxiv.org/abs/2008.01774). The overall structure of the proposed AI system is illustrated in the figure below. In this repository, we include the implementation for COVID-GMIC, COVID-GMIC-DRC, and COVID-GBM. We also provide an example script which trains and evalutes our models using randomly generated labels and images.

<p align="center">
  <img width="650" height="900" src="https://github.com/nyukat/COVID-19_prognosis/blob/master/system_overview_full.png">
</p>

## Prerequisites

* Python >= 3.6
* PyTorch >= 1.1.0
* torchvision >= 0.2.2
* NumPy >= 1.14.3
* imageio >= 2.4.1
* pandas >= 0.22.0
* opencv-python >= 3.4.2
* matplotlib >= 3.0.2
* LightGBM >= 2.3.1


## License

Please refer to [this file](https://github.com/nyukat/COVID-19_prognosis/blob/master/LICENSE) for details about the license of this repository.

## How to run the code

You need to first install conda in your environment. **Before running the code, please run `pip install -r requirements.txt` first.** Once you have installed all the dependencies, please navigate to the project directory `cd /path/to/COVID-19_prognosis` and run `export PYTHONPATH=$(pwd):$PYTHONPATH`. 

If you want to use our pretrained models and verify them for your own code and data, then this Jupyter [notebook](https://github.com/nyukat/COVID-19_prognosis/blob/master/notebooks/example-notebook.ipynb) contains code that performs model inference on example chest X-ray images and clinical variables data. It runs COVID-GMIC, COVID-GBM, and COVID-DRC.

If you want to train your own models based on our paper, we also provide a script that trains and evaluates models using synthetic data. It is almost plug-and-play, as you need to write your own data loader and tweak the parameters. Here is the command to execute the example script: 

`python covid19_prognosis/run_covid_model.py --device-type gpu --output-path output`

The following variables defined in `run_covid_model.py` can be modified as needed:
* `--seed`: The random seed that will be used throughout the script (default=random.randint(0, 99999)).
* `--output-path`: The directory where predictions will be saved (default=None).
* `--device-type`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu' (default="cpu", choices=['gpu', 'cpu']).
* `--gpu-number`: GPU number when multiple GPUs are available (default=0).
* `--drc`: whether to run for the outcome classification task or the DRC task (default=False).



## Reference

If you found this code useful, please cite our paper:

**An artificial intelligence system for predicting the deterioration of COVID-19 patients in the emergency department**\
Farah E. Shamout, Yiqiu Shen, Nan Wu, Aakash Kaku, Jungkyu Park, Taro Makino, Stanisław Jastrzębski, Jan Witowski, Duo Wang, Ben Zhang, Siddhant Dogra, Meng Cao, Narges Razavian, David Kudlowitz, Lea Azour, William Moore, Yvonne W. Lui, Yindalon Aphinyanaphongs, Carlos Fernandez-Granda, Krzysztof J. Geras\
arXiv:2008.01774, 2020.
    
    @article{shamout2020an, 
    title={An artificial intelligence system for predicting the deterioration of COVID-19 patients in the emergency department},
        author={Shamout, Farah E and Shen, Yiqiu and Wu, Nan and Kaku, Aakash and Park, Jungkyu and Makino, Taro and Jastrzębski, Stanisław and Witowski, Jan and Wang, Duo and Zhang, Ben and Dogra, Siddhant and Cao, Meng and Razavian, Narges and Kudlowitz, David and Azour, Lea and Moore, William and Lui, Yvonne W. and Aphinyanaphongs, Yindalon and Fernandez-Granda, Carlos and Geras, Krzysztof J},
        journal={arXiv:2008.01774},
        year={2020}}


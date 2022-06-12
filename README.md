# oadat-evaluate

[![launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/projects/firat.ozdemir/oa-armsim/sessions/new?autostart=1)

This project provides a multitude of tools around the neural network experiments presented in Lafci, B., Ozdemir, F., Dean-Bean, X.L., Razansky D., and Perez-Cruz, F. (2022). OADAT: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing.  
Namely, included tools allow to  
- load all presented pretrained models 
- evaluate a given pretrained model for a single sample
- train the presented models from scratch for each task
- batch evaluate a given serialized model (pretrained or your own) on corresponding OADAT test set for all presented metrics.
  
For more information on downloading datasets proposed within the scope of this work (MSFD, SWFD, SCD), check [github.com/berkanlafci/oadat](https://github.com/berkanlafci/oadat).

___
## Load presented pretrained models:  

A demo file showcasing how to load and evaluate any of the pretrained models is shown in file [demo.ipynb](notebooks/load_pretrained_models.ipynb).
The repository contains a sample for each task for each dataset for which one can see how to evaluate a pretrained model for a given sample without needing to download the whole dataset.   
This can also be checked on an interactive [renku session](https://renkulab.io/projects/firat.ozdemir/oa-armsim/sessions/new?autostart=1).   
The sample images are created using function [save_single_sample_for_each_task_to_repo](src/utils.py), which selects 1000th item from test set.


___
## Train models from scratch

We provide two scripts to train models from scratch, [one for image translation experiments (src/train_translation.py)](src/train_translation.py) and [another for semantic segmentation experiments (src/train_segmentation.py)](src/train_segmentation.py).  

One needs to download the corresponding dataset proposed within the scope of this work (MSFD, SWFD, SCD) in order to train from scratch. 
Next, modify the following attributes within ExpSetup class to fit your needs:  
- datasets_parent_dir: Directory where the corresponding dataset is located. Please do not rename the dataset files.   
- task_str: String variable that identifies the experiment to be trained for. Full list of experiments are   
`['msfd_lv128,li', 'swfd_lv128,li', 'swfd_lv128,sc', 'swfd_ss32,sc', 'swfd_ss64,sc', 'swfd_ss128,sc', 'scd_ss32,vc', 'scd_lv128,vc', 'scd_ss64,vc', 'scd_ss128,vc', 'scd_lv128,li', 'scd_lv256,ms', 'seg_ss32,vc', 'seg_ss128,vc', 'seg_ss64,vc', 'seg_lv128,li', 'seg_lv128,vc', 'seg_vc,vc']`
- logdir: Path to the directory where you want model checkpoints, logs during training, final serialized model after training and everything else to be saved.

___
## Evaluate a serialized model

We provide two scripts to evaluate a given serialized model (can be either one of the pretrained models we provide or a custom serialized model you provide), [one for image translation experiments (src/eval_translation.py)](src/eval_translation.py) and [another for semantic segmentation experiments (src/eval_segmentation.py)](src/eval_segmentation.py).  

One needs to download the corresponding dataset proposed within the scope of this work (MSFD, SWFD, SCD) in order to train from scratch. 
Next, modify the following attributes under main to fit your needs:  
- datasets_parent_dir: Directory where the corresponding dataset is located. Please do not rename the dataset files.   
- task_str: String variable that identifies the experiment to be trained for. Full list of experiments are   
`['msfd_lv128,li', 'swfd_lv128,li', 'swfd_lv128,sc', 'swfd_ss32,sc', 'swfd_ss64,sc', 'swfd_ss128,sc', 'scd_ss32,vc', 'scd_lv128,vc', 'scd_ss64,vc', 'scd_ss128,vc', 'scd_lv128,li', 'scd_lv256,ms', 'seg_ss32,vc', 'seg_ss128,vc', 'seg_ss64,vc', 'seg_lv128,li', 'seg_lv128,vc', 'seg_vc,vc']`
- fname_out: Path to the file where you want the computed metrics to be saved to.   

When evaluating one of the provided pretrained networks, this is sufficient. 
However, when evaluating a custom serialized model, you need to uncomment the following two lines and provide: 
- path_serialized_model: Path to the directory where the serialized model is saved to.

___
## Requirements

This project uses Tensorflow. We tested it to work for TF 2.7 and 2.8. 

___
## Citation  

TBA


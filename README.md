# oadat-evaluate

[![Launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/projects/firat.ozdemir/oadat-evaluate/sessions/new?autostart=1) 
[![arXiv](https://img.shields.io/badge/Preprint-arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.08612)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) 
[![Data](https://img.shields.io/badge/Data-Research%20Collection-blue)](https://www.research-collection.ethz.ch/handle/20.500.11850/551512)

This project provides a multitude of tools around the neural network experiments presented in Berkan Lafci, Firat Ozdemir, Xosé Luís Deán-Ben, Daniel Razansky, and Fernando Perez-Cruz. OADAT: Experimental and synthetic clinical optoacoustic data for standardized image processing. arXiv preprint arXiv:2206.08612, 2022.   
Namely, included tools allow to  
- load all presented pretrained models 
- evaluate a given pretrained model for a single sample
- train the presented models from scratch for each task
- batch evaluate a given serialized model (pretrained or your own) on corresponding OADAT test set for all presented metrics
- read any sample from the provided dataset, OADAT.
  
___

## Accessing the dataset: OADAT 

For information on downloading datasets proposed within the scope of this work (MSFD, SWFD, SCD), check [github.com/berkanlafci/oadat](https://github.com/berkanlafci/oadat).  
Dataset will be publicly available soon: [DOI for dataset files](https://www.research-collection.ethz.ch/handle/20.500.11850/551512).

___
## Load presented pretrained models:  

A demo file showcasing how to load and evaluate any of the pretrained models is shown in file [demo.ipynb](notebooks/load_pretrained_models.ipynb).
The repository contains a sample for each task for each dataset for which one can see how to evaluate a pretrained model for a given sample without needing to download the whole dataset.   
This can also be checked on an interactive [renku session](https://renkulab.io/projects/firat.ozdemir/oadat-evaluate/sessions/new?autostart=1).   
The sample images are created using function [save_single_sample_for_each_task_to_repo](src/utils.py), which selects 1000th item from test set.  

Sample script to load a pretrained model:
```python
import utils
mpm_obj = utils.Manage_Pretrained_Models()
task_str = 'seg_ss32,vc'
model = mpm_obj.load_model(task_str=task_str)
```


___
## Train models from scratch

We provide two scripts to train models from scratch, [one for image translation experiments (src/train_translation.py)](src/train_translation.py) and [another for semantic segmentation experiments (src/train_segmentation.py)](src/train_segmentation.py).  

One needs to download the corresponding dataset proposed within the scope of this work (MSFD, SWFD, SCD) in order to train from scratch. 
Next, modify the following attributes within ExpSetup class to fit your needs:  
- datasets_parent_dir: Directory where the corresponding dataset is located. Please do not rename the dataset files.   
- task_str: String variable that identifies the experiment to be trained for. Full list of experiments are   
`['msfd_lv128,li', 'swfd_lv128,li', 'swfd_lv128,sc', 'swfd_ss32,sc', 'swfd_ss64,sc', 'swfd_ss128,sc', 'scd_ss32,vc', 'scd_lv128,vc', 'scd_ss64,vc', 'scd_ss128,vc', 'scd_lv128,li', 'scd_lv256,ms', 'seg_ss32,vc', 'seg_ss128,vc', 'seg_ss64,vc', 'seg_lv128,li', 'seg_lv128,vc', 'seg_vc,vc', 'swfd_ss32,ms', 'swfd_ss64,ms', 'swfd_ss128,ms', 'seg_swfd_ss32,ms', 'seg_swfd_ss64,ms', 'seg_swfd_ss128,ms', 'seg_swfd_ss32,sc', 'seg_swfd_ss64,sc', 'seg_swfd_ss128,sc', 'seg_swfd_sc,sc', 'seg_swfd_ms,ms', 'seg_msfd_ss32,ms', 'seg_msfd_ss64,ms', 'seg_msfd_ss128,ms', 'scd_ss32,ms', 'scd_ss64,ms', 'scd_ss128,ms', 'seg_ss32,ms', 'seg_ss64,ms', 'seg_ss128,ms', 'seg_swfd_lv128,ms', 'seg_swfd_lv128,sc', 'seg_msfd_lv128,ms', 'msfd_ss32,ms', 'msfd_ss64,ms', 'msfd_ss128,ms']`
- logdir: Path to the directory where you want model checkpoints, logs during training, final serialized model after training and everything else to be saved.

Sample script to train an image translation model from scratch for experiment `swfd_ss128,sc`:  
```python
import train_translation
datasets_parent_dir = '/data/oadat' # assuming datasets downloaded here.
task_str = 'swfd_ss128,sc'
logdir = '/trained_models/oadat_swfd_ss128,sc'
args = train_translation.ExpSetup(datasets_parent_dir=datasets_parent_dir, task_str=task_str, logdir=logdir)
train_translation.train(args)
```
Sample script to train semantic segmentation model from scratch for experiment `seg_ss64,vc`:  
```python
import train_segmentation
datasets_parent_dir = '/data/oadat' # assuming datasets downloaded here.
task_str = 'seg_ss64,vc'
logdir = '/trained_models/oadat_seg_ss64,vc'
args = train_segmentation.ExpSetup(datasets_parent_dir=datasets_parent_dir, task_str=task_str, logdir=logdir)
train_segmentation.train(args)
```

___
## Evaluate a serialized model

We provide two scripts to evaluate a given serialized model (can be either one of the pretrained models we provide or a custom serialized model you provide), [one for image translation experiments (src/eval_translation.py)](src/eval_translation.py) and [another for semantic segmentation experiments (src/eval_segmentation.py)](src/eval_segmentation.py).  

One needs to download the corresponding dataset proposed within the scope of this work (MSFD, SWFD, SCD) in order to train from scratch. 
Next, modify the following attributes under main to fit your needs:  
- datasets_parent_dir: Directory where the corresponding dataset is located. Please do not rename the dataset files.   
- task_str: String variable that identifies the experiment to be trained for. Full list of experiments are   
`['msfd_lv128,li', 'swfd_lv128,li', 'swfd_lv128,sc', 'swfd_ss32,sc', 'swfd_ss64,sc', 'swfd_ss128,sc', 'scd_ss32,vc', 'scd_lv128,vc', 'scd_ss64,vc', 'scd_ss128,vc', 'scd_lv128,li', 'scd_lv256,ms', 'seg_ss32,vc', 'seg_ss128,vc', 'seg_ss64,vc', 'seg_lv128,li', 'seg_lv128,vc', 'seg_vc,vc', 'swfd_ss32,ms', 'swfd_ss64,ms', 'swfd_ss128,ms', 'seg_swfd_ss32,ms', 'seg_swfd_ss64,ms', 'seg_swfd_ss128,ms', 'seg_swfd_ss32,sc', 'seg_swfd_ss64,sc', 'seg_swfd_ss128,sc', 'seg_swfd_sc,sc', 'seg_swfd_ms,ms', 'seg_msfd_ss32,ms', 'seg_msfd_ss64,ms', 'seg_msfd_ss128,ms', 'scd_ss32,ms', 'scd_ss64,ms', 'scd_ss128,ms', 'seg_ss32,ms', 'seg_ss64,ms', 'seg_ss128,ms', 'seg_swfd_lv128,ms', 'seg_swfd_lv128,sc', 'seg_msfd_lv128,ms', 'msfd_ss32,ms', 'msfd_ss64,ms', 'msfd_ss128,ms']`
- fname_out: Path to the file where you want the computed metrics to be saved to.   

When evaluating one of the provided pretrained networks, this is sufficient. 
However, when evaluating a custom serialized model, you need to uncomment the following two lines and provide: 
- path_serialized_model: Path to the directory where the serialized model is saved to.


Sample script to evaluate a pretrained translation model (same logic applies to evaluating segmentation model):
```python
import utils
import eval_translation
mpm_obj = utils.Manage_Pretrained_Models()
task_str = 'swfd_lv128,li'
datasets_parent_dir = '/data/oadat' # assuming datasets downloaded here.
fname_out = '/trained_models/oadat_swfd_ss128,sc/eval.p'
model = mpm_obj.load_model(task_str=task_str)
eval_translation.eval(model, task_str, datasets_parent_dir, fname_out)
```

Sample script to evaluate any serialized (custom) model for a segmentation experiment (same logic applies to evaluating image translation model):
```python
import utils
import eval_segmentation
mpm_obj = utils.Manage_Pretrained_Models()
task_str = 'seg_ss64,vc'
datasets_parent_dir = '/data/oadat' # assuming datasets downloaded here.
fname_out = '/trained_models/oadat_seg_ss64,vc/eval.p'

path_serialized_model = '/trained_models/oadat_seg_ss64,vc/serialized_model_step_80000'
model = tf.keras.models.load_model(path_serialized_model, compile=False)
eval_segmentation.eval(model, task_str, datasets_parent_dir, fname_out)
```
___

## Use data loaders to read a sample from datasets

We provide a data loader class to read from datasets, whether it is to train a neural network model or simply analyzing the datasets. 
Sample script to read a desired index from desired dataset variables ([also available as a notebook](notebooks/read_data.ipynb)):  
```python
datasets_parent_dir = '/data/oadat' # assuming datasets downloaded here.
fname_dataset = 'SWFD_semicircle_RawBP.h5' ## example for SWFD semi circle dataset
fname_h5 = os.path.join(datasets_parent_dir, fname_dataset)
inds = None # if not None, generator will be limited to the provided dataset indices.
in_key = 'sc,ss32_BP' # example for semi circle sparse 32 images 
out_key = 'sc_BP' # example for semi circle array images
gen = generators.Generator_Paired_Input_Output(fname_h5=fname_h5, inds=inds, in_key=in_key, out_key=out_key, shuffle=True)
x, y = gen[42] # returns (in_key, out_key) tuple for 42th index in the dataset.
```
___
## Requirements

This project uses Tensorflow.  
We tested it to work for python 3.9 and TF 2.7 & 2.8. 

___
## Citation  

Please cite to this work using the following Bibtex entry:
```
@article{lafci2022oadat,
  doi = {10.48550/ARXIV.2206.08612},
  url = {https://arxiv.org/abs/2206.08612},
  author = {Lafci, Berkan and Ozdemir, Firat and Deán-Ben, Xosé Luís and Razansky, Daniel and Perez-Cruz, Fernando},
  title = {{OADAT}: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  journal={arXiv preprint arXiv:2206.08612},
}
```

___
## OA-armsim

You can find the source code for synthetic acoustic pressure map creation at [oa-armsim](https://renkulab.io/gitlab/firat.ozdemir/oa-armsim).

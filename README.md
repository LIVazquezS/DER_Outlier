# PhysNet DER modified

This is a new version of the PhysNet DER code for working with PES. The code computes energy, forces, charges, and dipole.
The model also computes the following two modified versions of DER:

- **DER-Lipzchitz**: This version uses a modified loss function that includes a term for Lipzchitz correction. 
                     It is adapted from [here](https://github.com/deargen/MT-ENet). The original reference can be found 
                     [here](https://arxiv.org/abs/2112.09368).
- **DER-Multidimensional**: This version uses a multidimensional prior, the Normal Inverse Wishart ([NIW](https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution))
                            distribution. It is adapted from [here](https://github.com/avitase/mder). 
                            The original reference can be found [here](https://arxiv.org/abs/2104.06135).


Additionally, the code includes the basic version of DER, which is the original [PhysNet DER](https://github.com/LIVazquezS/PhysNet_DER).

  
## To Do:
- [] Debug the code

## Using PhysNet Torch

### Requirements

- Python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.8
- PyTorch <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 1.10
- Torch-ema <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.3
- TensorBoardX <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4


## Using Physnet-DER
### Setting up the environment

We recommend to use [ Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) for the creation of a virtual environment. 

Once in miniconda, you can create a virtual environment called *physnet_torch* from the `.yml` file with the following command

``` 
conda env create --file environment.yml
```
 
To activate the virtual environment use the command:

```
conda activate physnet_torch
```

## Running the code

For training the model, you can use the following command:

```python run_train.py @input.inp ```

**NOTE**: You must define the type of DER that is wished to used. The options are: `simple`, `lipz`, and `MD`. 
If this is not defined, the code will not run.

## Contact

This is still a work in progress, if you have questions or problems please contact:

L.I.Vazquez-Salazar, email: luisitza.vazquezsalazar@unibas.ch

## Reference

- Vazquez-Salazar, L. I.; Boittier, E. D.; Meuwly, M. Uncertainty quantification for
predictions of atomistic neural networks. Chem. Sci. 2022, 13 (44), 13068-13084. DOI: [10.1039/D2SC04056E](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc04056e)
- Vazquez-Salazar, L.I.; Käser S.; Meuwly, M. Outlier-Detection for Reactive Machine Learned Potential Energy Surfaces. 
arXiv e-prints 2024, arXiv: [arXiv:2402.17686](https://arxiv.org/abs/2402.17686).
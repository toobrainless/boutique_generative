# boutique_generative

This is a repository for BHW2 (Generative models for images) homework of HSE DL Course. Project includes DCGAN model written in PyTorch and trained on [AnimeFaces](https://www.kaggle.com/datasets/splcher/animefacedataset), [CelebFaces](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

## Project structure

Repository is structured in the following way.

-   `main.py`: script for training model.

-   `config.py`: configs for model training.

-   `models.py`: code for Generator and Discriminator of DCGAN.

-   `trainer.py`: code for training model.

-   `utils.py`: basic utils including `load_dataset` for dataloaders structuring, `create_gif` for gif creation and etc. 

## Training

To train model run the following script:

```
python3 main.py
```

## Testing

To evaluate model run the following script:

```
python3 generate.py --checkpoint_path <path_to_checkpoint>
```


## Authors

-   Artur Gimranov

## Credits

Example was taken from [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
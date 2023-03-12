<h2 align="center">Convolutional VAE</h2>
<hr>

A convolutional variational autoencoder based on a MNIST dataset.

### Train
To run the training process run ``train.py`` with given parameters:

* **--epochs** - number of epochs; 
* **--batch_size** - batch size;
* **--latent_dims** - size of a latent dimension;
* **--save_path** - path to save model;

``
python train.py --epochs=10 --batch_size=128 --save_path=./model_vae.pt --latent_dims=10
``

### Run

To run a model and view the result run ``inference.py`` with given parameters:

* **--model_path** - path to a model; 
* **--save_path** - path to save an image;
* **--start_num** - number to start from;
* **--end_num** - number to finish;
* **--latent_dims** - size of a latent dimension of a model.

```
python inference.py --model_path=./model_vae.pt --save_path=./misc/two_to_five.jpg --latent_dims=10 --start_num=2 --end_num=5
```

<div style="display: flex; flex-direction: row;">
    <img src="./misc/two_to_five.jpg" width="600">
    <img src="./misc/six_to_one.jpg" width="600">
</div>

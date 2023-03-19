# Number of workers for dataloader
WORKERS = 2

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE = 64

# Number of channels in the training images. For color images this is 3
CHANNELS = 3  # nc

# Size of z latent vector (i.e. size of generator input)
VECTOR_SIZE = 100  # nz

# Size of feature maps in generator
GEN_FEATURES = 64

# Size of feature maps in discriminator
DISC_FEATURES = 64

# Learning rate for optimizers
LEARNING_RATE = 0.0002

# Beta1 hyperparam for Adam optimizers
BETA1 = 0.5

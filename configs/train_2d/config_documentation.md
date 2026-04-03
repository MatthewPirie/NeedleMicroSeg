
## extractor
Defines how raw images and masks are spatially processed before training.

- name
  - full_frame_resize: resizes full image to a fixed resolution

- kwargs.out_hw
  - [H, W]: output height and width after resizing


## normalization
Controls how image intensities are normalized.

- name
  - zscore_per_image: normalize each image using its own mean/std
  - null: no normalization applied


## augmentations
Controls data augmentation during training.

- enabled
  - List of augmentations to apply
  - Options:
    - flip: random horizontal/vertical flips
    - translate: random spatial shifts
    - rotate_scale: random rotation and scaling
    - elastic: elastic deformation
    - noise: additive Gaussian noise
    - blur: Gaussian smoothing
    - shift: intensity shift
    - scale: intensity scaling
    - contrast: contrast adjustment
  - [] = no augmentations

- kwargs
  - Optional parameters for augmentations
  - If not provided, default values are used


## data
Controls dataset splitting.

- split_id
  - Identifier for train/validation split
  - Example: kfold_cv_fold-0


## train
Controls training behavior and optimization.

- batch_size
  - Number of samples per batch

- epochs
  - Number of full training passes

- steps_per_epoch
  - Number of optimizer updates per epoch (can be independent of dataset size)

- log_every
  - Frequency (in steps) for logging training metrics

- optimizer
  - adam: adaptive optimizer (default)
  - sgd: stochastic gradient descent with momentum

- lr
  - Initial learning rate

- weight_decay
  - L2 regularization strength

- momentum
  - Momentum term (used only for SGD)

- lr_scheduler
  - none: constant learning rate
  - cosine: cosine decay over training
  - polynomial: polynomial decay

- min_lr
  - Minimum learning rate (used for cosine scheduler)

- poly_power
  - Controls decay curve shape (used for polynomial scheduler)


## loss
Controls loss function weighting.

- w_bce
  - Weight for binary cross-entropy loss

- w_dice
  - Weight for Dice loss

- batch_dice
  - true: compute Dice over batch
  - false: compute Dice per sample

# Brain MRI Segmentation with a U-Net CNN Architecture

The goal of this project is to train a neural network to segment (i.e. outline) target regions in images. In particular, the model will be trained to segment abnormalities in brain MRIs.

This type of problem falls within the realm of computer vision, where the industry standard tools are convolutional neural networks (CNNs). CNNs use filters, small windows that perform linear calculations as they slide over the entire image to learn features of the image ("convolving" with the image). Groups of filters working together form the layers of the CNN, where different layers learn different features due to their position within the network and varying random initializations. As in all neural networks, CNNs must also have layers separated by nonlinear activation functions to introduce nonlinearity into the model.

A first neural network approach might be to "flatten" the pixels of the (often three-channel) image into a long one-dimensional vector then build a fully-connected network consisting of connections between all pixels or neurons. By contrast, the filters of CNNs connect neurons only to others nearby resulting in fewer parameters to be optimized (contained within the filters) and faster training. In addition, CNNs use of filters preserves the spatial structure of the multi-dimensional image. On the other hand, a single convolution layer will consist of many filters or channels to enable learning which can lead to high memory requirements, especially if small filters are used. While larger filters could help, it has been found to be better to use pooling layers to reduce output volumes or 1x1 convolution layers to reduce channels.

This leads to a "contracting" network architecture where image sizes decrease with successive layers while the number of feature channels/filters may increase. This is problematic for the task at hand since we require a final pixel-wise classification, that is, our last output must be the same size as the original input image. The "U-Net" architecture solves this by following the initial contracting path of the architecture with a symmetric expanding path that uses upsampling operations to increase image resolution. This project implements upsampling with transposed convolutions which introduces additional learnable parameters and should make the model more robust, although upsampling can also be done with no additional parameters through "unpooling" operations. The contracting path learns contextual information while the expanding path - which includes concatenation with features of the same size from the contracting path - enables localization. Many CNN architectures end with a fully connected layer but the U-Net disposes of fully connected layers entirely, opting instead for a 1x1 convolution for the final classification layer, making for a very concise overall structure.

<img src="https://i.postimg.cc/zGg6t9Pq/Screenshot-2024-02-06-093305.png" width="550">

Some other components of this U-Net implementation include the following:
* Image augmentation: A challenge common to medical imaging problems is the lack of large datasets. Additional data can be simulated by applying transformations and deformations to the available images. Another (probably better) point of view is that augmentation serves as a form of regularization to prevent overfitting to a relatively small dataset and allows the model to learn invariance to these augmentations.
* Adam optimization: Most machine learning models are optimized numerically with some form of the gradient descent algorithm since analytic optimization is rarely possible. Gradient descent would ideally be done with the entire training dataset to update parameters but this would be very computationally expensive. In practice, models are typically updated with iterations of "mini-batches" from the training set, known as batch gradient descent (BGD). Adam optimization adjusts BGD to correct for some of the downsides of numerical approximation and the use of mini-batches.
Gradient descent updates often have some directions that progress in large steps while others update in small steps, resulting in undesirable oscillations and slow progress. Adam is a type of "adaptive" optimization that sums previous updates to scale new updates in each direction in order to encourage more even progress.
Adam also uses update history to accumulate "momentum" from previous BGD updates in order to keep parameters heading in the right direction and correct for noisiness present in single BGD updates.
* Loss function = BCE + (1-DICE): A standard loss function in binary image segmentation that sums classical binary cross entropy loss (BCE) and DICE loss (1-DICE). DICE represents the Dice score or coefficient which measures the ratio of intersection (overlap) to union of the target and predicted segmentation regions.

<img src="https://i.postimg.cc/zBVnp10j/Screenshot-2024-02-15-101254.png" width="1000">

Improvements in progress:
* Learning rate scheduling
* More sophisticated models (attention mechanisms?): implementing a U-Net was a good learning experience but many improvements have been made since 2015
* Transfer learning with pretrained models
 
Sources:
* Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-NET: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/pdf/1505.04597.pdf)
* Kaggle [(dataset)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data)
* [Carvana Image Masking Challenge–1st Place Winner’s Interview](https://www.medium.com/kaggle-blog/carvana-image-masking-challenge-1st-place-winners-interview-78fcc5c887a8)
* CS231n Deep Learning for Computer Vision, Stanford Univ.



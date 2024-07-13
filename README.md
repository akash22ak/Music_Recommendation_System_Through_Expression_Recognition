# Music Recommendation Through Facial Recognition

This project involves designing and developing a CNN model to classify facial expressions into the following categories: 0 - angry, 1 - disgust, 2 - fear, 3 - happy, 4 - sad, 5 - surprise, and 6 - neutral, based on images. The model then recommends music accordingly. The framework used for this project is Keras with TensorFlow.

## Introduction

Facial expressions are a primary means by which people express their emotions. Music has long been recognized for its ability to influence an individual's mood. By capturing and recognizing a person's emotional state and suggesting suitable songs, this system aims to calm the user's mind and provide a pleasant experience.

## Libraries and Frameworks

The CNN model is developed using Keras, a framework of [TensorFlow](https://www.tensorflow.org). Additional Python libraries such as Numpy, Pandas, and Matplotlib are used for image processing, data manipulation, data analysis, and graph plotting.

## Dataset

### Facial Expression Recognition

The dataset comprises 48x48 pixel grayscale images of faces, with a total of 35,887 images.

A bar graph depicting the number of images for each emotion:

![dataset](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f1.png?raw=true)

The dataset is divided into a training set containing 80% of the images and a validation set containing 20%.

## Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a type of deep learning model particularly effective in analyzing grid-like data, such as images, by utilizing convolution. CNNs have significantly advanced the field of computer vision, achieving impressive results in tasks like image classification, object detection, and image segmentation.

### Convolutional Layers

CNNs use convolutional layers with multiple learnable filters or kernels. Each filter performs a convolution operation, sliding across the input data, computing dot products with local patches of the input. This extracts spatial features and learns local patterns. The result is a set of feature maps that capture different aspects of the input data.

### Pooling Layers

Pooling layers downsample the feature maps from convolutional layers, reducing spatial dimensions. Max pooling, the most common pooling operation, selects the maximum value within a defined window. Pooling captures the most important features, reduces computational complexity, and provides some translation invariance.

### Fully Connected Layers

After convolutional and pooling layers, CNNs often include fully connected layers, which connect every neuron in one layer to every neuron in the next. These layers learn high-level representations by combining features extracted by convolutional layers. The final fully connected layer produces the output, which can be used for classification, regression, or other tasks.

![238127907-e07d1fd2-0b02-4693-8d04-1921bbfae833](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f2.png?raw=true)

## Model Architecture

- Four cascading convolutional layers (Conv2D) and pooling layers (MaxPooling2D) are added, specifying the number of filters and the activation function (ReLU) used.
- Three fully connected layers are defined, decreasing in units until a 7-member layer is obtained, classified using a SoftMax activation.
- "DropOut" layers are added to remove hidden nodes, improving model accuracy.

![model](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f3.png?raw=true)

### Data Augmentation

Data augmentation artificially increases the dataset size by creating modified copies of existing data. Images are modified with random rotation, shear, translations, and noise. This technique makes the model more robust by teaching it to handle irregularities in data.

![data_aug](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f4.png?raw=true)

## Results

- The training accuracy reached around 72%, and validation accuracy plateaued at around 64% by 40 epochs.

![accuracy](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f5.png?raw=true)

![loss](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f6.png?raw=true)

- This indicates "overfitting" – where the model fails to improve its training accuracy, usually due to data imbalance as seen in the first dataset plot.
- Data augmentation helps overcome this by creating new images for the model to process, thereby improving accuracy.

The output is the emotion detected from the facial expression captured by the input camera feed.

![happy](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f7.png?raw=true)

![sad](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f8.png?raw=true)

### Song Recommendation

![sadsong](https://github.com/akash22ak/Music_Recommendation_System_Through_Expression_Recognition/blob/master/images/f9.png?raw=true)

## Conclusion

In conclusion, this project presents a successful approach for classifying emotions using convolutional neural networks (CNNs). The proposed model achieves good accuracy and specificity in identifying different facial expressions. Future work should address the limitations of this study and explore additional methodologies to further enhance the model's performance.

## References

1. [Kaggle Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
2. [Research Paper](https://www.researchgate.net/publication/351056923_Facial_Expression_Recognition_Using_CNN_with_Keras)

# Traffic-sign-Detections-Using-YOLOV3-Detector
## AI > Computer Vision > Object Detection project
Yolo is an algorithm that uses convolutional neural networks for object detection.
In comparison to recognition algorithms, a detection algorithm does not only predict class labels, but detects locations of objects as well.
### 1. Model hyperparameters: we define some configurations for Yolo.
Batch normalization: Almost every convolutional layer in Yolo has batch normalization after it. It helps the model train faster and reduces variance between units (and total variance as well).
Leaky ReLU: Leaky ReLU is a slight modification of ReLU activation function. The idea behind Leaky ReLU is to prevent so-called "neuron dying" when a large number of activations become 0.
Anchors: Anchors are sort of bounding box priors, that were calculated on the COCO dataset using k-means clustering. We are going to predict the width and height of the box as offsets from cluster centroids. The center coordinates of the box relative to the location of filter application are predicted using a sigmoid function.
### 2. Model definition: It refered to the official ResNet implementation in Tensorflow in terms of how to arange the code.
### 3. Batch norm and fixed padding: It's useful to define batch_norm function since the model uses batch norms with shared parameters heavily. Also, same as ResNet, Yolo uses convolution with fixed padding, which means that padding is defined only by the size of the kernel.
### 4. Feature extraction: Darknet-53: For feature extraction Yolo uses Darknet-53 neural net pretrained on ImageNet. Same as ResNet, Darknet-53 has shortcut (residual) connections, which help information from earlier layers flow further. We omit the last 3 layers (Avgpool, Connected and Softmax) since we only need the features.
### 5. Convolution layers: Yolo has a large number of convolutional layers. It's useful to group them in blocks.
### 6. Detection layers: Yolo has 3 detection layers, that detect on 3 different scales using respective anchors.\
For each cell in the feature map the detection layer predicts n_anchors * (5 + n_classes) values using 1x1 convolution.
For each scale we have n_anchors = 3. 5 + n_classes means that respectively to each of 3 anchors we are going to predict 4 coordinates of the box, its confidence score (the probability of containing an object) and class probabilities.
### 7. Upsample layer: In order to concatenate with shortcut outputs from Darknet-53 before applying detection on a different scale, we are going to upsample the feature map using nearest neighbor interpolation.
### 8. Non-max suppression: The model is going to produce a lot of boxes, so we need a way to discard the boxes with low confidence scores. Also, to avoid having multiple boxes for one object, we will discard the boxes with high overlap as well using non-max suppresion for each class.
### 9. Final model class: Finally, let's define the model class using all of the layers described previously.
### 10. Utility functions: Here are some utility functions that will help us load images as NumPy arrays, load class names from the official file and draw the predicted boxes.
### 11. Converting weights to Tensorflow format: Now it's time to load the official weights. We are going to iterate through the file and gradually create tf.assign operations.
## Running the model: Now we can run the model using some sample images.
Detections:
Testing the model with IoU (Interception over Union ratio used in non-max suppression) threshold and confidence threshold both set to 0.5.
## Video processing
I also applied the same algorithm to video detections.

# CSE 490/599 G1 Homework 2 #

Welcome friends,

For the second assignment we'll be looking at Convolutional Neural Networks (CNNs), the things that make computer vision work. 
In this homework you will implement your own Convolution and Max Pool layers, and eventually will even create a ResNet style layer.
You will again be working on MNIST, but now you should be able to get up to 99% accuracy! 
We're also going to work on CIFAR, and even ImageNet, more challenging datasets on real images.

We have provided some tests for checking your implementation. Feel free to write more. But the tests for Conv and MaxPool should be sufficient for those layers.
To run the tests from the outermost directory, simply run
```bash
pytest test/hw2_tests
```
Or to run them for an individual file (for example test_conv_layer), run
```bash
pytest test/hw2_tests/test_conv_layer.py
```

## Rules ##
1. You may not use PyTorch or any other deep learning package in parts 1-5 of the homework. Only Numpy and Numba are allowed. Functions like numpy.matmul are fine to use.
1. You may not use prewritten convolution functions like `scipy.ndimage.convolve` or `cv2.filter2D`. I also think if you did use them, they wouldn't cover all the cases like padding and stride that we need, so it's best just to ignore them.
1. You may only modify the files we mention (those in the [submit.sh](submit.sh) script). We will not grade files outside of these.
1. You may not change the signatures or return types of `__init__`, `forward`, `backward`, or `step` (`forward_numba` and `backward_numba` are ok to modify as you see fit) in the various parts or you will fail our tests. You may add object fields (e.g. `self.data`) or helper functions within the files.
1. Undergrads partners only need to turn in a single homework, but you must put both partner's NetIDs in partners.txt comma separated on a single line.
    Example: `studenta,studentb`
1. You may talk with others about the homework, but you must implement by yourself (except partners).
1. Those not working with a partner should leave partners.txt blank.
1. Feel free to test as you go and modify whatever you want, but we will only grade the files from [submit.sh](submit.sh).

## 1. Convolutions ##

If you check out [nn/layers/conv_layer.py](nn/layers/conv_layer.py) you will see a pretty empty file. You should implement the convolution algorithm as described in class.
To make everyone's lives easier, we are only asking for you to cover a few cases of convolutions. If you want to do more, you're more than welcome, but as soon as you pass the provided tests, you should be good to go.
Specifically, you can make the following assumptions/simplifications:
1. You will "technically" be implementing the cross-correlation, but everyone calls it convolution.
1. The input to the layer will always be 4-Dimensional `(batch, input_channels, height, width)`.
1. The output should be `(batch, output_channels, output_height, output_width)`
1. Convolutional kernels will always be square.
1. Padding is defined as how many rows to add to ALL 4 sides of the input images. e.g. If an input has shape `(2, 3, 100, 100)` and there is padding of 3, the padded input should be of shape `(2, 3, 106, 106)`.
1. You should always use the padding rule `padding = (kernel_size - 1) // 2`, and should always pad with zeros. This rule is nice because if your stride is 1 and the kernel-size is odd, the output will be the same shape as the input.
1. When computing output sizes, you should discard incomplete rows if the stride puts it over the edge. e.g. `(2, 3, 5, 5)` input, 3 kernels of size 2x2 and stride of 2 should result in an output of shape `(2, 3, 2, 2)`.
1. You can expect sane sizes of things and don't have to explicitly error check. e.g. The kernel size will never be larger than the input size or larger than the stride.


## 2. Max Pooling ##
This should be very similar to convolution except instead of multiplying, you do a max. The same rules and expectations apply as in the conv layer.

### 3 Flatten Layer ###
We will need this to be able to feed the output of a convolution layer (which outputs `b x c x h x w`)into a linear layer (which expects `b x c`).
This should be very straightforward to implement. All you need to do is reshape the output (have a look at the np.reshape) function.
The `backward` should also be quite straightforward.


### Training the network ###
With this done, and using our code from last time, we can now train a full convolutional network! Open [hw2/main.py](hw2/main.py). We have already provided code to train and test.
The network here is actually a very famous network called LeNet. If you would have written this in 1995, you'd be up for a Turing award this year.

Much of the training code will also look familiar from last time.
You can run it by calling
```bash
cd hw2
python main.py
```
After 1 epoch, you should see about 95% test accuracy. After 5 epochs, you should see about ______% accuracy.


## 5. Improvements ##



TO WRITE





We can apply numerous improvements over the simple neural network from earlier. After implementing each improvement, you will need to modify [hw1/main.py](hw1/main.py) to use the new network or optimizer.

### 5.1 Momentum SGD ###
Our normal SGD update with learning rate η is:

```math 
w \Leftarrow w - \eta * \frac{\partial L}{\partial w}
```

With weight decay λ we get:
```math 
w \Leftarrow w - \eta * \left( \frac{\partial L}{\partial w} + \lambda w \right)
```

With momentum we first calculate the amount we'll change by as a scalar of our previous change to the weights plus our gradient and weight decay:
    
```math 
\Delta w \Leftarrow m * \Delta w_{prev} +  \left( \frac{\partial L}{\partial w} + \lambda w \right)
```

Then we apply that change:
   ```math 
w \Leftarrow w - \eta \Delta w
``` 

The MomentumSGDOptimizer class will need to keep a history of the previous changes in order to compute the new ones. 

Using momentum should give significantly faster and better convergence. After a single epoch, you should see about 90% accuracy. After 10 epochs, you should see about 95% accuracy.

### 5.2 Leaky ReLU Layer ###
Implement the Leaky ReLU function `LeakyReLU(x) = x if x > 0, slope * x if x <= 0`. We will define the gradient at 0 to be like the negative half. 
(Note, we know we changed this from what it originally said. If you already implemented the version we originally said, you will pass our tests, but this new version is more correct).

You may see LeakyReLU defined in other places as `LeakyReLU(x) = max(x, slope * x)` but what happens if slope is > 1 or negative? Be careful of this trap in your implementation.
You can implement this using either Numpy or Numba, whichever you find easier. 

### 5.3 Parameterized ReLU (PReLU) Layer ###
Implement the PReLU function where the leaky slope is a learned parameter. Again, we will define the gradient at 0 to be like the negative half. 
(Note, we know we changed this from what it originally said. If you already implemented the version we originally said, you will pass our tests, but this new version is more correct).


For more information, see https://arxiv.org/pdf/1502.01852.pdf

PReLU can be either one value per channel (which we will assume is dimension 1 of the input) or one slope for the entire layer.
Thus, the `size` input will be an integer >= 1. 

You can implement this using either Numpy or Numba, whichever you find easier. 
If you implement using Numba, we recommend not using the `parallel=True` flag to ensure that your gradient computations do not overwrite each other. 
However we will give +1 extra credit (and +1 deep learning street cred) if you submit a parallel version.

## 6. PyTorch ##
Navigate to: https://colab.research.google.com/

and upload the iPython notebook provided: `homework1_colab.ipynb`

Complete the notebook to train a PyTorch model on the MNIST dataset.

## 7. Short answer ##
Answer these questions and save them in a file named `hw1/short_answer.pdf`.
1. Play around with different Leaky ReLU slopes. What is the best slope you could find? What happens if you set the slope > 1? What about slope < 0. Theoretically, what happens if you set slope = 1?
2. Set PReLU to take 1 slope per layer. After 20 epochs, what were your PReLU slopes? Does this correspond with what you found in question 1?
3. If you add more layers and more epochs, what accuracy can you reach? Can you get to 99%? What is your best network layout?

## Turn it in ##

First `cd` to the `hw1` directory. Then run the `submit.sh` script by running:

```bash
bash submit.sh
```

This will create the file `submit.tar.gz` in your directory with all the code you need to submit. The command will check to see that your files have changed relative to the version stored in the `git` repository. If it hasn't changed, figure out why, maybe you need to download your ipynb from google?

Submit `submit.tar.gz` in the file upload field for Homework 1 on Canvas.

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import matplotlib.pyplot as plt
%matplotlib
plt.style.use('ggplot')
plt.ion()
%autoindent
import sys

srng = RandomStreams()

def floatX(X):
    '''Stuff variable into numpy array with theano float datatype'''
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    '''Initialize weights intelligently with random values'''
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    '''ReLU, rectify linear Tensor function in theano'''
    return T.maximum(X, 0.)

def softmax(X):
    '''Typical softmax/MNL function, where X = theta*x, size is #classes'''
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    '''Implement random dropout of some neurons during training'''
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    '''Adaptive learning rate, with decay rho and initial learning rate lr'''
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    '''This is the actual layering of the CNN'''
    # a ReLU on 3x3 blocks, zero pad if block partially empty
    l1a = rectify(conv2d(X, w, border_mode='full'))
    # pool, so top scoring from a group of 4, within each feature
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)
    # a ReLU on 3x3 blocks, only use on complete parts of intermediate feature block (no zero pad)
    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)
    #next layer
    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    #next layer
    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)
    #next layer
    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

# load the data
trX, teX, trY, teY = mnist(onehot=True)

# reshape training set to be 2-dimensional
# negative is to get ordering right (rather than mirror image)
trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

# let's take a look at one of these images
## looking at the first image, with another index for the channel (even though we only have 1 hear)
plt.pcolor(trX[1][0], cmap=plt.cm.gray_r)


# theano 4-d tensor of floats, note this is exactly what trX is
X = T.ftensor4()
# theano matrix of floats, so 2d with some special methods
Y = T.fmatrix()

# These will change depending on our convolutional business

# We will extract 32 features (neurons), on 1 channel, from a 3x3 grid of the image
w = init_weights((32, 1, 4, 4))

# Let's make 64 features, on 32 channels (note that prev layer gave us this many), again on 3x3 
w2 = init_weights((64, 32, 4, 4))

# crank it up to 128 features, on 64 channels (notice a pattern?), still 3x3
w3 = init_weights((128, 64, 4, 4))

# This is fully connected again, 128 features from above
# 3 * 3 just happens to be the size of the features at this step, after all the conv2d and maxpool
w4 = init_weights((128 * 2 * 2, 625))

# final fully connected layer, we have 625 input neurons, and need weights for each of the 10 possible digits
w_o = init_weights((625, 10))

# This sets up the model graph and some vars for noise, which are the internal neurons
noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
# No dropping neurons...so this will be used for prediction?
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
# this makes predictions
y_x = T.argmax(py_x, axis=1)

# here's our cost function, how wrong we are, we'll use this to train
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
# this is what gets updated
params = [w, w2, w3, w4, w_o]
# graph the updates
updates = RMSprop(cost, params, lr=0.001)

# compile the training and prediction functions
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# after we've trained the above, we can do prediction...using the shared weights
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# finally run the model
alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(100):
    # minibatch of size128
    num_mini_batch = np.ceil(len(trX)/128)
    for start, end in zip(list(range(0, len(trX), 128)), list(range(128, len(trX), 128))):
        cost = train(trX[start:end], trY[start:end])
        if (start/128) % np.ceil(num_mini_batch/20) == 0:
            print(alph[i%len(alph)],end='')
            sys.stdout.flush()
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))

# let's look at some of the weights
ww = w.get_value()
# ok, this is 32 features, and we can see the contribution from each pixel in the grid
len(ww)
# subplot size
sx,sy = (8,4)
f, con = plt.subplots(sx,sy, sharex='col', sharey='row')
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolor(ww[yy*4+xx][0], cmap=plt.cm.gray_r)

# let's go up one level and look at that
ww2 = w2.get_value()
# subplot size
sx,sy = (8,8)
f, con = plt.subplots(sx,sy, sharex='col', sharey='row')
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolor(ww2[yy*8+xx][0], cmap=plt.cm.gray_r)


# examine final inputs
ww = w_o.get_value()
plt.figure()
plt.pcolor(ww.T, cmap=plt.cm.gray_r)


# what if we do the google dream thing, fix the output (or a particular layer), and update the initial 'image' input in order to see what it thinks a '7' is, or whatever
# image will need to be a proper theano tensor variable

def init_rand_img(shape):
    '''Initialize weights intelligently with random values'''
    return theano.shared(floatX(np.random.rand(*shape)))

img = init_rand_img([1,1,28,28])
start = img.get_value()[0][0]

def dream_model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    '''This is the actual layering of the CNN'''
    # a ReLU on 3x3 blocks, zero pad if block partially empty
    l1a = rectify(conv2d(X, w, border_mode='full'))
    # pool, so top scoring from a group of 4, within each feature
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)
    # a ReLU on 3x3 blocks, only use on complete parts of intermediate feature block (no zero pad)
    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)
    #next layer
    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    #next layer
    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)
    #next layer
    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

l1, l2, l3, l4, dream_pyx = dream_model(img, w, w2, w3, w4, 0, 0)

def ascend(img, digit, cost, lr = 0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost,wrt=img)
    updates = []
    for p, g in zip(img, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g)) # we add because we ascend!
    return updates

dream_cost = T.mean(T.nnet.categorical_crossentropy(dream_pyx, Y))
params = [img]
dream_updates = ascend(params,y_x,dream_cost,lr = 0.1)


dream = theano.function(inputs=[Y], outputs=img, updates = dream_updates, allow_input_downcast=True)

all_digits = [start]
for i in range(1,16):
    dream_digit = dream([trY[3]])
    all_digits.append(img.get_value()[0][0])

sx,sy = (4,4)
f, con = plt.subplots(sx,sy, sharex='col', sharey='row')
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolor(all_digits[yy*4+xx], cmap=plt.cm.gray_r)

# dream on certain layers next...instead of cost being py_x vs Y, cost should be...l4 or l3 or something


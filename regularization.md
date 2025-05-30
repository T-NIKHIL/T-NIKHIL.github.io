---
layout: page
permalink: /regularization/
show_excerpts: true
---

<p align='justify'>
    As scientists and engineers when we build a mathemetical model 
    to describe a process or system, we often start with the simplest
    explanation or model. This line of thinking is pervasive throughout
    science and engineering and goes by many names (Ex : Occam's razor).
    Simple models are not only more intuitive and interpretable, but are also
    more likely to generalize outside the domain it was built on. This guiding principle has found its way into the world of machine learning and led to devleopment of several 
    strategies to train machine learning models to generalize better. These strategies collectively fall under the name of <em>regularization</em>. In this tutorial we will be looking at
    parameter normed penalties. 
</p>

<p align="center">
    <img src="/assets/occams_razor.pdf" alt="Occam's Razor">
</p>

### Creating a toy dataset

<p align='justify'>
    Lets start by building a toy dataset with 2 variables
    ($x_1$ and $x_2$) and a dependent variable $y$ that 
    linearly depends on the two independent variables 
    as $y = 3x_1 + 2x_2 + N(0,0.5)$. Code to generate
    the dataset is shown below. 
</p>

```python
seed = 777
num_data = 10
w = np.array([[3, 2]]).reshape(-1, 1)

def f(w, x):
    np.random.noise(seed)
    random.noise(seed)
    gaussian_noise = np.random.normal(0, 1, size=x.shape[0]).reshape(-1, 1)
    return np.dot(x, w) + 0.5*gaussian_noise

np.random.seed(seed)
random.seed(seed)
x1 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)
x2 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)

features = np.hstack((x1, x2))
target = f(w, features)
dataset = np.hstack((features, target))
```

<p align='justify'>
    We can visualize the surface without the gaussian noise and
    is shown in red in the below figure. The black dots are the
    dataset we will use to train our model.
</p>

<iframe id="igraph" scrolling="yes" 
style="border:none; margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/linear_surface.html"
height="380" width="740"></iframe>

### Coding the model and loss function

Our linear model is $\hat{y} = w_1x_1 + w_2x_2$. 
Let's also define a loss function that will tell us how good our model
parameters $w_1$ and $w_2$ are. For this we will use
the mean squared error (MSE) loss function.
 
$$
J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

```python
class LinearModel():
    def __init__(self):
        self.w = None

    def forward(self, x, w, save=True):
        if save:
            self.w = w
        return np.dot(x, self.w)

def loss_fn(targets, preds):
    return np.mean((targets.reshape(-1, 1) - preds.reshape(-1, 1))**2)
```

Lets explore how this loss function looks like in 3D.
The dotted black line shows where the global minima exists for this surface.

<iframe id="igraph" scrolling="yes" 
style="border:none; margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/loss_surface.html" height="380" width="740"></iframe>

### Gradient Descent

Now lets write a simple code to do gradient descent to update
our model parameters.

```python
def dl_dw1(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 0].reshape(-1, 1)))

def dl_dw2(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 1].reshape(-1, 1)))

# Initial value to start optimization from
w_init = np.array([[-10, -10]], dtype=np.float32).reshape(-1, 1)
num_epochs = 40
# Learning rate
lr = 0.1

linear_model = LinearModel()
preds = linear_model.forward(dataset[:,:-1], w_init)
preds_hist = preds.reshape(-1, 1)
w_hist = deepcopy(w_init)
w_new = deepcopy(w_init)

for epoch in range(num_epochs):
    w_grads = np.array([dl_dw1(dataset, preds), dl_dw2(dataset, preds)]).reshape(-1, 1)
    w_new -= lr*w_grads
    preds = linear_model.forward(train_dataset[:,:-1], w_new)
    w_hist = np.hstack((w_hist, w_new))
    preds_hist = np.hstack((preds_hist, preds.rehape(-1, 1)))
```

<iframe id="igraph" scrolling="yes" 
style="border:none; 
margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/GD_path_no_l1_reg.html" height="380" width="740"></iframe>

<p align='justify'>
    The left plot in the above figure traces out the parameter trajectory 
    in the parameter space and the right plot shows the fitting 
    with the final model parameters. The algorithm is able to 
    find parameters close to the true parameters. 
</p>

### Increasing number of variables in the dataset

<p align='justify'>
    Now consider the case where instead of just fitting 2 variables,
    we have to fit a linear model with many more variables than data
    available to train the model. For this lets double the number of variables while still maintaining the number of datapoints.
</p>

```python
num_data = 10
w_true = np.array([[3, 2, 0, 0]]).reshape(-1, 1)

np.random.seed(seed)
random.seed(seed)
x1 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)
x2 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)
x3 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)
x4 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)

features = np.hstack((x1, x2))
target = f(w, features)
dataset = np.hstack((features, target))

w_init = np.array([[-10, -10, -10, -10]], dtype=np.float32).reshape(-1, 1)
num_epochs = 600
lr = 0.1

linear_model = LinearModel()
preds = linear_model.forward(dataset[:,:-1], w_init)
preds_hist = preds.reshape(-1, 1)
loss = loss_fn(dataset[:, -1], preds)
loss_hist = loss.reshape(-1, 1)
w_hist = deepcopy(w_init)
w_new = deepcopy(w_init)

for epoch in range(num_epochs):
    w_grads = np.array([dl_dw1(dataset, preds, w_new, lambda_reg), 
                        dl_dw2(dataset, preds, w_new, lambda_reg),
                        dl_dw3(dataset, preds, w_new, lambda_reg),
                        dl_dw4(dataset, preds, w_new, lambda_reg)]).reshape(-1, 1)
    w_new -= lr * w_grads
    preds = linear_model.forward(dataset[:,:-1], w_new)
    preds_hist = np.hstack((preds_hist, preds.reshape(-1, 1)))
    loss = loss_fn(dataset[:, -1], preds)
    loss_hist = np.hstack((loss_hist, loss.reshape(-1, 1)))
    w_hist = np.hstack((w_hist, w_new))
```

<p align="center">
    <img src="/assets/blogs/regularization/loss_curve_for_4_feats.pdf" alt="Loss Curve for 4 features">
</p>

<p align='justify'>
    One thing we immediately observe is that the number of epochs has
    increased by almost 20x by just adding 2 more independent variables to
    the dataset. In the loss plot we see a steep drop in the first 50 epochs and 
    then a gradual decline over the next 300 epochs. We
    see by 300 epochs, the model has found that the first 2 variables
    are the major contributors to the target variable prediction. Over the 
    next 300 epochs, the loss curve starts to plateau and the model
    refines its parameter estimates.
</p>

<p align='justify'>
    <b>Exercise 1</b> : Tune the learning rate, number of epochs and
    initial parameter guesses to see if you can bring the loss down
    further. Are you able to find better parameter estimates ?
</p>

<p align='justify'>
    After playing around for a while, you will realize that the model
    gets close to the true parameter set but never actually reaches it.
    Why does this happen ? This is because the model we have is too complex
    for the dataset we are trying to fit. The model tries to use all available 
    variables given to it but doesn't realize that only the first 2 variables
    are sufficient to explain the target variable. After thinking for a while
    you might come up with a solution to add a constraint on the model
    parameters to be as small as possible. How can we mathemetically formalize
    this constraint ? This is exactly with parameter norm penalties do. 
</p>

### Vector Norms

<p align='justify'>
    Wolfram Mathworld defines a norm as a quantity that in an abstract
    sense defines length, size or extent of an object. We can think of our
    model parameters as a rank-1 tensor (i.e vector). For any real number
    p >= 1, the p-norm of a vector is defined as below, where 'n' is the
    number of model parameters and 'p' defines the type of norm we are using.
</p>

$$
||w||_p = (|x_1|^p + |x_2|^p + |x_3|^p \dots + |x_n|^p)^{\frac{1}{p}}
$$

<p align='justify'>
    Substituting p=1, we get the 1-norm also called $L^1$ norm or Manhattan norm.
    Substituting p=2, we get the famous 2-norm also called $L^2$ norm or Euclidean norm. The $L^\infty$ norm or maximum norm is the limit of the $L^p$
    norm as $p \to \infty$. The $L^\infty$ norm is interesting as only points at the corner of the square satisfy the constraint. The below plot shows the level sets of $L^1$, $L^2$, $L^3$ and $L^\infty$ norms in 2D.
</p>

<p align="center">
    <img src="/assets/blogs/regularization/norms.pdf" alt="L1 to Linf norm">
</p>

### Modifying loss function to include parameter norm penalties

<p align='justify'>
    Lets start by adding the $L^1$ norm constraint to our loss function and stick with the 2 parameter case so we can visualize
    the loss surface. The loss function we arrived at is the same loss
    function used in least absolute shrinkage and selection operator (LASSO).
</p>

$$
J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda(|w_{1}| + |w_{2}|)
$$

<p align='justify'>
    The $\lambda$ parameter controls the strength of the
    $L^1$ norm penalty. This is called as a hyperparameter
    and requires manual tuning as we will see below.
    For now lets fix the value to 0.5.
</p>

```python
# Updating loss and grad functions to include L1 norm penalty ...
def loss_fn(targets, preds, w, lambda_reg):
    return np.mean((targets.reshape(-1, 1) - preds.reshape(-1, 1))**2) + lambda_reg*(np.abs(w[0, 0]) + np.abs(w[1, 0]))

def dl_dw1(dataset, preds, w, lambda_reg):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 0].reshape(-1, 1))) + lambda_reg*np.sign(w[0, 0])

def dl_dw2(dataset, preds, w, lambda_reg):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 1].reshape(-1, 1))) + lambda_reg*np.sign(w[1, 0])
```

### Gradient Descent with modified loss function

<iframe id="igraph" scrolling="yes" 
style="border:none; 
margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/loss_surface_with_l1_reg.html" height="380" width="740"></iframe>

<p align='justify'>
    Where the MSE loss surface and $L^1$ norm constraint meet, defines the set of 
    solutions that satisfy the constraint. Now the optimizer must find where along
    this 1-D curve the loss is minimized. Sticking with the same num_epochs and 
    learning rate as in the case without $L^1$ norm constraint lets run our gradient 
    descent code and visualize the parameter 
    trajectory and final optimized model parameters.
</p>

<iframe id="igraph" scrolling="yes" 
style="border:none; 
margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/GD_path_with_l1_reg.html" height="380" width="740"></iframe>

<p align='justify'>
    Not suprisingly, the model performs even worse. The surface 
    defined by the $L^1$ norm penalty actually pushes the model
    parameters back and prevents it from reaching the global minimum.
</p>

<p align='justify'>
    <b>Exercise 3</b> : Vary the $\lambda$ parameter and see how it affects
    the loss surface and the parameter trajectory on the loss surface.
</p>

<p align='justify'>
    <b>Exercise 4</b> : Instead of the $L^{1}$ norm parameter penalty, change
    the loss function and grad functions to include the $L^{2}$ norm
    penalty. The loss function you arrive at is the one used in Ridge
    Regression. Visualize how this loss surface looks like. Run gradient descent and see what model parameters you arrive at.
</p>

### Revisiting the 4 variable dataset

<p align='justify'>
    Now that we have some intuition of what is happening, lets go 
    back to fitting the model to the 4D dataset. 
</p>

<p align="center">
    <img src="/assets/blogs/regularization/loss_curve_for_4_feats_w_l1_reg.pdf" alt="loss curve with 4 feats">
</p>

<p align='justify'>
    We see that the loss curve plateaus very early
    on with high regularization. The parameters
    weighting the last 2 features are the smallest
    when $\lambda=1$ but are still not completely 'zeroed' out yet.
    Large values of $\lambda$ also impact the parameters for feature 1 and 2,
    increasing the bias in the model.
    $\lambda=0.1$ offers a good balance
    between accuracy and minimizing influence
    of features 3 and 4. 
</p>

<p align='justify'>
    A simple strategy to 'zero' out small parameters is thresholding.
    We can define a threshold or range such that any parameter less
    than the threshold or within the range will be set to 0. 
    This is how the sequentially thresholded least squares (STLSQ) algorithm works.
    The parameter vector obtained at the end of training is a sparse vector. 
    This optimization algorithm is used in sparse identification of 
    nonlinear dynamics (SINDy) [1].
</p>

### Vector norms with $0 \leq p < 1$

<p align='justify'>
    TODO
</p>

<p align='justify'>
[1] S.L. Brunton, J.L. Proctor, & J.N. Kutz, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, Proc. Natl. Acad. Sci. U.S.A. 113 (15) 3932-3937, https://doi.org/10.1073/pnas.1517384113 (2016).
</p>
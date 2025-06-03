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
    as $y = 3x_1 + 2x_2 + N(0,1)$. Code to generate
    the dataset is shown below. 
</p>

```python
seed = 777
num_data = 10
w_true = np.array([[3, 2]]).reshape(-1, 1)

def f(w, x):
    np.random.seed(seed)
    random.seed(seed)
    gaussian_noise = np.random.normal(0, 1, size=x.shape[0]).reshape(-1, 1)
    return np.dot(x, w) + gaussian_noise

np.random.seed(seed)
random.seed(seed)
x1 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)
x2 = np.round(np.random.uniform(-1, 1, num_data), 1).reshape(-1, 1)

# Standardize the features
x1_mean = np.mean(x1, axis=0)
x1_std = np.std(x1, axis=0)
x1_standardized = (x1- x1_mean) / x1_std
x2_mean = np.mean(x2, axis=0)
x2_std = np.std(x2, axis=0)
x2_standardized = (x2 - x2_mean) / x2_std

features = np.hstack((x1_standardized, x2_standardized))
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

def mse(targets, preds):
    return np.mean((targets.reshape(-1, 1) - preds.reshape(-1, 1))**2)
```
<p align='justify'>
    Lets explore how this loss function looks like in 3D.
</p>

<iframe id="igraph" scrolling="yes" 
style="border:none; margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/mse_surface.html" height="380" width="740"></iframe>

<p align='justify'>
    We can see that the mse loss surface is ellisoidal in shape. 
    The dotted black line represents the minima of the loss surface.
    The color of the surface represents the value of the MSE loss.
</p>

### Gradient Descent

Now lets write a simple code to do gradient descent to update
our model parameters.

```python
# This is the code to calculate gradient of the MSE function wrt w1
def dl_dw1(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 0].reshape(-1, 1)))

# This is the code to calculate gradient of the MSE function wrt w2
def dl_dw2(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 1].reshape(-1, 1)))

# Initial value to start optimization from
w_init = np.array([[-5, -5]], dtype=np.float32).reshape(-1, 1)
num_epochs = 10
# Learning rate
lr = 0.1

linear_model = LinearModel()
preds = linear_model.forward(dataset[:,:-1], w_init)
preds_hist = preds.reshape(-1, 1)
mse_loss = mse(dataset[:, -1], preds)
mse_loss_hist = mse_loss.reshape(-1, 1)
w_hist = deepcopy(w_init)
w_new = deepcopy(w_init)

for epoch in range(num_epochs):
    w_grads = np.array([dl_dw1(dataset, preds), dl_dw2(dataset, preds)]).reshape(-1, 1)
    w_new -= lr*w_grads
    preds = linear_model.forward(train_dataset[:,:-1], w_new)
    preds_hist = np.hstack((preds_hist, preds.rehape(-1, 1)))
    mse_loss = mse(dataset[:, -1], preds)
    mse_loss_hist = np.hstack((mse_loss_hist, mse_loss.reshape(-1, 1)))
    w_hist = np.hstack((w_hist, w_new))
    
```

<iframe id="igraph" scrolling="yes" 
style="border:none; 
margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/GD_path_no_l1_reg.html" height="380" width="740"></iframe>

<p align='justify'>
    The left plot in the above figure traces out the parameter trajectory 
    in the parameter space and the right plot shows the fitting 
    with the final model parameters. The algorithm is able to 
    find parameters close to the true parameters. Since we started
    with good initial guesses we were able to converge fast to the
    global minima. 
</p>

<p align='justify'>
    <b>Exercise 1</b> : Find the number of epochs it would take
    to converge to the global minima starting from the adjacent 
    corner of the loss surface.
</p>

### Increasing number of variables in the dataset

<p align='justify'>
    Now consider the case where instead of just fitting 2 variables,
    we have to fit a linear model with many more variables but the
    dataset does not sufficiently sample the observation space.
    For this lets double the number of variables while still maintaining the number of datapoints. 
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

x1_mean = np.mean(x1, axis=0)
x2_mean = np.mean(x2, axis=0)
x3_mean = np.mean(x3, axis=0)
x4_mean = np.mean(x4, axis=0)

x1_std = np.std(x1, axis=0)
x2_std = np.std(x2, axis=0)
x3_std = np.std(x3, axis=0)
x4_std = np.std(x4, axis=0)

x1_standardized = (x1 - x1_mean) / x1_std
x2_standardized = (x2 - x2_mean) / x2_std
x3_standardized = (x3 - x3_mean) / x3_std
x4_standardized = (x4 - x4_mean) / x4_std

features = np.hstack((x1_standardized, x2_standardized, x3_standardized, x4_standardized))
target = f(w, features)
dataset = np.hstack((features, target))

def mse(targets, preds):
    return np.mean((targets.reshape(-1, 1) - preds.reshape(-1, 1))**2)

def dl_dw1(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 0].reshape(-1, 1)))

def dl_dw2(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 1].reshape(-1, 1))) 

def dl_dw3(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 2].reshape(-1, 1)))

def dl_dw4(dataset, preds):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 3].reshape(-1, 1)))

class LinearModel():
    def __init__(self):
        self.w = None

    def forward(self, x, w, save=True):
        if save:
            self.w = w
        return np.dot(x, self.w)

w_init = np.array([[-5, -5, -5, -5]], dtype=np.float32).reshape(-1, 1)
num_epochs = 200
lr = 0.1

linear_model = LinearModel()
preds = linear_model.forward(dataset[:,:-1], w_init)
preds_hist = preds.reshape(-1, 1)
mse_loss = loss_fn(dataset[:, -1], preds)
mse_loss_hist = loss.reshape(-1, 1)
w_hist = deepcopy(w_init)
w_new = deepcopy(w_init)

for epoch in range(num_epochs):
    w_grads = np.array([dl_dw1(dataset, preds), dl_dw2(dataset, preds),
                        dl_dw3(dataset, preds), dl_dw4(dataset, preds)]).reshape(-1, 1)
    w_new -= lr * w_grads
    preds = linear_model.forward(dataset[:,:-1], w_new)
    preds_hist = np.hstack((preds_hist, preds.reshape(-1, 1)))
    mse_loss = mse(dataset[:, -1], preds)
    mse_loss_hist = np.hstack((mse_loss_hist, mse_loss.reshape(-1, 1)))
    w_hist = np.hstack((w_hist, w_new))
```

<p align="center">
    <img src="/assets/blogs/regularization/loss_curve_with_4_feats_with_l1_0.pdf" alt="Loss Curve for 4 features">
</p>

<p align='justify'>
    One thing we immediately observe is that the number of epochs has
    increased by 20x by just adding 2 more independent variables to
    the dataset. In the $log(MSE)$ plot above we can see that the loss
    steeply drops in the first 20 epoch and then plateaus after 100 epochs.
</p>

<p align='justify'>
    <b>Exercise 2</b> : Tune the learning rate, number of epochs and
    initial parameter guesses to see if you can bring the loss down
    further. Are you able to find better parameter estimates ?
</p>

<p align='justify'>
    After playing around for a while, you will realize that the model
    finds reasonable parameter values for the first two parameters but
    cannot completely "zero" out the last two parameters. Why does this
    happen ? This is because of the way the model has been coded 
    we require the model to use all the available features and thus during the
    optimization process while the model tries to limit the influence of 
    spurious features, it cannot completely ignore them. This challenge
    is exactly what sparse regression methods try to solve.
    One solution to this problem is to add a constraint during the optimization
    process that forces the model to learn parameters that are as small as possible. This is exactly with parameter norm penalties do. 
    How can we mathemetically formalize this constraint ? 
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
    Substituting p=2, we get the famous 2-norm also called $L^2$ norm or Euclidean norm. 
    The $L^\infty$ norm or maximum norm is the limit of the $L^p$ norm as $p \to \infty$.
    In 3D, the level sets of $L^\infty$ norm are a cube and in larger dimensions it forms
    a hypercube. The below plot shows the level sets of $L^1$, $L^2$, $L^3$ and $L^\infty$ norms in 2D.
</p>

<p align="center">
    <img src="/assets/blogs/regularization/level_sets_of_convex_norms.pdf" alt="L1 to Linf norm">
</p>

### Modifying loss function to include parameter norm penalties

<p align='justify'>
    Lets start by adding the $L^1$ norm parameter penalty to our loss function and stick with the 2 parameter case so we can visualize
    the loss surface. The loss function we arrived at is the same loss
    function used in least absolute shrinkage and selection operator (LASSO).
</p>

$$
J = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda(|w_{1}| + |w_{2}|)
$$

<p align='justify'>
    The $\lambda$ parameter controls the strength of the
    $L^1$ norm parameter penalty. This is called as a hyperparameter
    and requires manual tuning as we will see below.
    For now lets fix the value to 1.0.
</p>

```python
def mse(targets, preds):
    return np.mean((targets.reshape(-1, 1) - preds.reshape(-1, 1))**2)

def l1_norm(w, lambda_reg=0.1):
    return lambda_reg * np.sum(np.abs(w))

def dl_dw1(dataset, preds, w, lambda_reg=0.1):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 0].reshape(-1, 1))) + lambda_reg*np.sign(w[0, 0])

def dl_dw2(dataset, preds, w, lambda_reg=0.1):
    return np.mean(2*(dataset[:, -1].reshape(-1, 1) - preds.reshape(-1, 1))*(-dataset[:, 1].reshape(-1, 1))) + lambda_reg*np.sign(w[1, 0])
```

### Gradient Descent on MSE loss surface with $L^1$ norm penalty

<iframe id="igraph" scrolling="yes" 
style="border:none; 
margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/mse_plus_l1_norm_surface.html" height="380" width="740"></iframe>

<p align='justify'>
    Where the MSE loss surface and $L^1$ norm constraint meet, defines the set of 
    solutions that satisfy the constraint. Now the optimizer tries to find the minima
    while balancing the constraint to have parameters as small as possible.
    Sticking with the same num_epochs and learning rate as in the case without 
    the $L^1$ norm constraint lets run our gradient descent code and 
    visualize the parameter trajectory and final optimized model parameters.
</p>

<iframe id="igraph" scrolling="yes" 
style="border:none; 
margin-bottom:0" seamless="seamless" src="/assets/blogs/regularization/GD_path_with_l1_reg.html" height="380" width="740"></iframe>

<p align='justify'>
    Comparing to the case without the $L^1$ norm penalty we 
    see that the parameter norm penalty works and the 
    parameters have reduced slightly, thus increasing the MSE loss.
</p>

<p align='justify'>
    <b>Exercise 3</b> : Vary the $\lambda$ parameter and see how it affects
    the loss surface and the parameter trajectory on the loss surface.
</p>

<p align='justify'>
    <b>Exercise 4</b> : Instead of the $L^{1}$ norm parameter penalty, change
    the loss function and grad functions to include the $L^{2}$ norm
    penalty. The loss function you arrive at is the one used in Ridge
    Regression. Visualize how this loss surface looks like. 
    Run gradient descent and see what model parameters you arrive at.
</p>

### Revisiting the 4 variable dataset

<p align='justify'>
    Now that we have some intuition of what is happening, lets go 
    back to fitting the model to the 4D dataset with the $L^1$ norm
    parameter penalty. 
</p>

<p align="center">
    <img src="/assets/blogs/regularization/loss_curve_with_4_feats_with_l1_reg.pdf" alt="loss curve with 4 feats">
</p>

<p align='justify'>
    We see that the loss curve plateaus rapidly in all the cases,
    indicating that the model finds the global minima basin early on.
    With high values of regularization, all the model parameters are
    significantly constrained and we get a model with poor predictive power.With $\lambda=0.01$ we get a model that has parameters similar to the 
    case without $L^1$ parameter penalty. $\lambda=1.0$ offers a good balance
    of predictive accuracy while minimizing influence of features 3 and 4. 
</p>

<p align='justify'>
    A simple strategy to 'zero' out small parameters is thresholding.
    We can define a threshold or range such that any parameter less
    than the threshold or within a range will be set to 0. 
    This is how the sequentially thresholded least squares (STLSQ) algorithm works.
    The parameter vector obtained at the end of training is a sparse vector. 
    This optimization algorithm is one of the optimization algorithms
    used in sparse identification of nonlinear dynamics (SINDy) [1].
</p>

### Vector norms with $0 \leq p < 1$

<p align='justify'>
    <img src="/assets/blogs/regularization/level_sets_of_nonconvex_norms.pdf" alt="nonconvex norms">
</p>

<p align='justify'>
[1] S.L. Brunton, J.L. Proctor, & J.N. Kutz, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, Proc. Natl. Acad. Sci. U.S.A. 113 (15) 3932-3937, https://doi.org/10.1073/pnas.1517384113 (2016).
</p>

### Appendix

```python
# Basic plotly code for visualization
w1, w2 = np.arange(-5, 10, 0.1), np.arange(-5, 10, 0.1)
lambda_reg = 1.0

w1_grid, w2_grid = np.meshgrid(w1, w2)
mse_surface = np.zeros_like(w1_grid)
l1_norm_surface = np.zeros_like(w1_grid)

for i in range(w1_grid.shape[0]):
    for j in range(w1_grid.shape[1]):
        preds = w1_grid[i, j] * dataset[:, 0] + w2_grid[i, j] * dataset[:, 1]
        mse_surface[i, j] = mse(dataset[:, -1], preds)
        w_arr = np.array([[w1_grid[i, j]], [w2_grid[i, j]]])
        l1_norm_surface[i, j] =  l1_norm(w_arr, lambda_reg)

# Create 3D scatter plot for loss surface
loss_surface = go.Figure()

# Add pred loss surface
loss_surface.add_trace(go.Surface(
    z=mse_surface,
    x=w1_grid,
    y=w2_grid,
    colorscale='Viridis',
    opacity=1.0,
    showscale=False,
    name=''
))

# Add the L1 norm surface
loss_surface.add_trace(go.Surface(
    z=l1_norm_surface,
    x=w1_grid,
    y=w2_grid,
    colorscale='reds',
    opacity=1.0,
    showscale=False,
    name=''
))

# Add a line at the global minimum point
z_min = np.nanmin(mse_surface)
z_max = np.nanmax(mse_surface)
z = np.linspace(z_min, z_max, 100)
loss_surface.add_trace(go.Scatter3d(
    x=[3]*len(z),
    y=[2]*len(z),
    z=z,
    mode='markers',
    marker=dict(
        size=1,
        color='black',
        opacity=1.0
    ),
    name=''
))

loss_surface.update_layout(
    scene=dict(
        xaxis_title='w1',
        yaxis_title='w2',
        zaxis_title='MSE Loss'
    ),
    xaxis=dict(
        tickvals = w1,
        ticktext = [f'{w}' for w in w1]
    ),
    yaxis=dict(
        tickvals = w2,
        ticktext = [f'{w}' for w in w2]
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    width=720,
    height=360
)
loss_surface.show()
loss_surface.write_html('loss_surface.html')
```
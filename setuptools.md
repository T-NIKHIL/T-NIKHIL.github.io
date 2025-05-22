---
layout: page
permalink: /setuptools/
show_excerpts: true
---

This guide serves as a short summary of some popular guides and documentation (referenced below) 
I have read when learning how to build a python package. Any views expressed in this article are my own.

<p align='justify'>
    Python packages provide an easy way for users to import the 
    necessary methods and classes in your code through the python <em>import</em> function.
    A built python package is stored as a <em>python wheel</em> (.whl) 
    which is a ZIP-format archive, containing all the files 
    necessary to install your python package. 
    <em>Wheels</em> are the latest standard for storing python packages, replacing <em>eggs</em>.
    Once a <em>wheel</em> is generated it can be distributed locally or 
    pushed to the <em>Python Package Index (PyPI)</em> which is global server 
    hosting python software. If you have ever done <em>pip install X</em>, 
    then <em>pip</em> would most likely be downloading from PyPI.
</p>


### Build Frontend vs Build Backend

<p align='justify'>
    Those familiar using <em>pip</em> should be aware that it
    cannot directly convert your code into a python package. 
    That is the job of a <em>build backend</em>.
    <em>pip</em> and <em>build</em> are examples of 
    what the community calls as <em>build frontends</em>, 
    tools that help install python packages.
    In the background these <em>build frontend</em> tools identify the 
    dependencies of the package you requested and attempt 
    to resolve any dependency conflicts among the dependencies. 
    They do so by downloading the <em>source distribution</em> for each dependency
    and calling the build backend on each to install the dependency 
    and check for conflicts. Tools like <em>build</em> create an environment
    with the appropriate python packages that will invoked by the 
    <em>build backend</em> tools (called setup dependencies), 
    during the installation of your python code (called runtime dependencies) 
    and finally during testing (called test dependencies).
</p>

<p align='justify'>
    At this point I should probably mention the subtle distinction
    between a <em>source</em> and <em>built distribution</em>. 
    A <em>source distribution</em> is an archive file (stored as .tar.gz) 
    that contains your source code and metadata. 
    It is not platform specific (ex:Windows, Linux) distribution. 
    While a <em>built distribution</em> is an archive file (stored as .whl)
    that is specific to your hardware, OS and python version 
    and can be run directly without having to install python. 
</p>

<p align='justify'>
    <em>Build backends</em> are responsible for converting your <em> source tree </em>
    - <em> the directory containing all files and folders relevant to your source code </em> -
    to either a <em>source</em> or <em>build distribution</em>. Examples of <em>build backends</em>
    are <em>flit-core, hatchling, maturin, steuptools</em> and <em>poetry</em>. 
    In this tutorial we will be focusing on using <em>setuptools</em>.
</p>

#### Step 1 : *pip install build* 

While we can invoke <em>setuptools</em> directly, it is easier and preferred 
to invoke it through the python <em>build</em> module. To start make sure 
you have <em>build</em> installed in your python environment type 
`pip install --upgrade build` in your terminal.

#### Step 2 : Structure your codebase

Now for the purpose of this demo we create a test_package directory with following layout.

```plaintext
test_package/
├── pyproject.toml
├── src/
│   └── test_package/
│       ├── __init__.py
│       └── math_operations.py
├── tests/
│   ├── test_add.py
│   ├── test_multiply.py
│   └── test_exp.py
├── LICENSE
├── README.md
└── test_package.yml
```

In this tutorial we will be focusing on <em>pyproject.toml</em> and the <em>src</em> directory. 
In math_operations.py lets define some simple math operations :

```python
# math_operations.py
import numpy

def add(x, y):
    return x+y

def multiply(x, y):
    return x*y

def exp(x):
    return np.exp(x)
```

The `__init__.py` is a file that is required for the test_package directory
to be registered as a module which allows us to write commands like : 
`from test_package.math_operations import add`. 
It can also can store information related to your source code such 
as authors, license and package version. Here is a sample `__init__.py` layout:

```python
# __init__.py
__author__ = "your_name"
__license__= "MIT'"
__version__= "0.0.1"
__all__ = ["add", "multiply", "exp"] 
```

The `__all__` entry is used to specify what all modules should be 
imported when a user does a wildcard import.

#### Step 3 : Write the *pyproject.toml* file

This file tells the <em>build frontend</em> tools, what <em>build backend</em> 
to use which in this case is setuptools. As a side note, for those familiar
with using <em>setup.py</em> or <em>setup.cfg</em> for configuring the build
of their <em>source tree</em> should note that this practice is slowly getting
depreciated. Please refer to this 
<a href="https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html"> blog post</a>
for more details. Below shows a very simple <em>pyproject.toml</em> file that 
covers the basic required items. For more advanced utilities I would refer you 
to the <em>setuptools</em> documentation. For instance this does not cover how to 
include tools like <em>pytest</em> unit testing and how to incorporate linting 
tools. When including these tools make sure to include a 
`[project.optional-dependencies]`section.

```python
# pyproject.toml
[build-system]
requires=["setuptools"]
build-backend="setuptools.build_meta"

[project]
name="test_package"
authors=[
    {name = "your_name", email = "your_email@xyz.com"},
]
description = "your_project_description"
readme="README.md"
license= {text = "MIT"}
requires-python=">=3.8"
keywords=["some_keywords", "separated_by_commas"]
dependencies = [
          "numpy",
]
dynamic=["version"]

[project.urls]
Homepage = "https://your_website.com"
Documentation = "https://readthedocs.org"
Repository = "https://github.com/me/your_git_repo.git"

[tool.setuptools.dynamic]
version={attr="test_package.__version__"}
```

#### Step 4 : Building the *wheel* and *source distribution* using *build*

Now go back to the directory containing <em>pyproject.toml</em> and in your terminal 
type `python -m build --wheel` to create a <em>built distribution</em> or 
type `python -m build --sdist` to create a <em>source distribution</em>. Both the 
<em>built</em> and <em>source distributions</em> are located under the <em>dist</em> directory.
The <em>wheel</em> will be listed with the package version number and python versions 
as `test_package-0.0.1-py3-none-any.whl`.

Once we have the <em>built distribution</em> we can use 
<em>pip</em> to install the package and its dependencies (which in this case is numpy). 
To do this go into the <em>dist</em> directory and type 
`pip install test_package-0.0.1-py3-none-any.whl` in your terminal.

Voilá ! you have now created your very own (and maybe first!) python package. 
You can now type python in your terminal and import the different math 
functions we have defined.

```terminal
user@computer python
>> from test_package import math_operations
>> math_operations.add(2, 3)
5
>> math_operations.exp(2)
7.38905609893065
```

References :\\
[1] https://packaging.python.org/en/latest/tutorials/packaging-projects/ \\
[2] https://setuptools.pypa.io/en/latest/userguide/quickstart.html \\
[3] Some GitHub repos you can refer to :  https://github.com/joreiff/rodeo, https://github.com/rxhernandez/ReLMM



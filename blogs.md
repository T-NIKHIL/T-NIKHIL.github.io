---
layout: page
title: Blogs
permalink: /blogs/
show_excerpts: true
---

An amalgam of how-to-use guides on software tools, 
explanations of topics I found interesting 
and some personal thoughts and views. I would love 
to hear your comments/suggestions on these topics.
Please reach out via email !

<ul class="post-list">
{%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  <li>
    <span class="post-meta">{{ '2024-08-26' | date: date_format }}</span>
    <h3>
      <a class="post-link" href="https://colab.research.google.com/drive/1CQ5z-zfUY3PQJgZRrk-Bd632hJ476Nih" style="color: #0055ff">
        PyTorch vs TensorFlow: A comparison
      </a>
    </h3>
    <p align='justify'>
      Alright, so now you know the basics of how neural networks work. You might be wondering which python framework 
      would be best for you to start building your own nets. This interactive google collab notebook will help you
      choose between the two most popular frameworks: PyTorch and TensorFlow.
    </p>
  </li>
<ul class="post-list">
{%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  <li>
    <span class="post-meta">{{ '2025-02-11' | date: date_format }}</span>
    <h3>
      <a class="post-link" href="/setuptools/index.html" style="color: #0055ff">
        Creating python packages using build and setuptools
      </a>
    </h3>
    <p align='justify'>
      You have done the hard work ! Now make it easy for others to use and build on your code by creating a Python package. Once done, your
      code is just a 'pip install' away from finding a new home !
    </p>
  </li>
<ul class="post-list">
{%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  <li>
    <span class="post-meta">{{ '2025-05-27' | date: date_format }}</span>
    <h3>
      <a class="post-link" href="/regularization/index.html" style="color: #0055ff">
        Building models in the low data regime : Part 1. Regularization
      </a>
    </h3>
    <p align='justify'>
      Lets face it ! As a data scientist, we all regularly encounter
      situations where after training a model on a dataset, we test it
      on new data, only to observe that the model performs poorly at the
      task at hand. This usually happens because the model is too complex 
      (i.e has too many parameters) than is required and so it attempts to
      memorize the training data rather than trying to learn the underlying
      patterns. Regularization encompasses a set of strategies to help 
      create models with better performance at test time, sometimes at the expense of performance during training. 
    </p>
  </li>

<hr style="border: 1px solid black;"/>



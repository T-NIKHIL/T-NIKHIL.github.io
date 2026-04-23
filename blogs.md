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
    <span class="post-meta">{{ '2026-04-22' | date: date_format }}</span>
    <h3>
      <a class="post-link" href="/msa/index.html" style="color: #0055ff">
        Searching for evolutionarily conserved sequences using 
        Multiple Sequence Alignment (MSA)
      </a>
    </h3>
    <p align='justify'>
      In this blogpost I explain how Multiple Sequence Alignment (MSA) 
      works using a toy example and how two different classes of algorithms
      implement MSA in practice.
    </p>
  </li>

<ul class="post-list">
{%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  <li>
    <span class="post-meta">{{ '2026-04-14' | date: date_format }}</span>
    <h3>
      <a class="post-link" href="/alphafold3/index.html" style="color: #0055ff">
        AlphaFold3 : Predicting Biomolecular Structures from Sequence
      </a>
    </h3>
    <p align='justify'>
      AlphaFold3 represents Google DeepMind's as of date latest 
      open source model for bio-molecular structure prediction.
      In this tech blog, I dive deep into how AlphaFold3 works,
      how does it perform on benchmark datasets and what are its
      limitations. 
    </p>
  </li>

<ul class="post-list">
{%- assign date_format = site.minima.date_format | default: "%b %-d, %Y" -%}
  <li>
    <span class="post-meta">{{ '2025-05-27' | date: date_format }}</span>
    <h3>
      <a class="post-link" href="/regularization/index.html" style="color: #0055ff">
        Regularize your models ! Part 1. Parameter Norm Penalties
      </a>
    </h3>
    <p align='justify'>
      In this tutorial we will look at a widely used strategy to 
      learn sparse model parameters by adding parameter norm
      penalties to the loss function.
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

<hr style="border: 1px solid black;"/>



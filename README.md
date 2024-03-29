# repo-compilation
Compilation of intresting and useful to me repos

LF AI & Data Foundation Interactive Landscape: https://landscape.lfai.foundation/

Table of Contents
=================

* [repo-compilation](#repo-compilation)
* [Table of Contents](#table-of-contents)
    * [CMD line interface](#cmd-line-interface)
    * [Visualization](#visualization)
    * [AutoML](#automl)
    * [Model debugging](#model-debugging)
    * [DL fast research and low level controll](#dl-fast-research-and-low-level-controll)
    * [DL frameworks for fast prototyping and pretrained models](#dl-frameworks-for-fast-prototyping-and-pretrained-models)
    * [NLP](#nlp)
    * [Audio processing](#audio-processing)
    * [CV](#cv)
    * [Time Series](#time-series)
    * [Recommendations](#recommendations)
    * [Distributed](#distributed)
    * [Data mining and data processing](#data-mining-and-data-processing)
    * [Enterprise ML and DL](#enterprise-ml-and-dl)
    * [Speed up](#speed-up)
    * [Testing](#testing)
    * [Python app creator helpers:](#python-app-creator-helpers)
    * [Automate](#automate)

### CMD line interface

- https://github.com/tiangolo/fastapi

    FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

- https://github.com/pallets/click - almost one liner

    Click is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary. It's the "Command Line Interface Creation Kit". It's highly configurable but comes with sensible defaults out of the box.

### Visualization

- https://github.com/apache/superset -  Apache Superset is a Data Visualization and Data Exploration Platform
    Superset provides:

    - An intuitive interface for visualizing datasets and crafting interactive dashboards
    - A wide array of beautiful visualizations to showcase your data
    Code-free visualization builder to extract and present datasets
    - A world-class SQL IDE for preparing data for visualization, including a rich metadata browser
    - A lightweight semantic layer which empowers data analysts to quickly define custom dimensions and metrics
    - etc.

- https://github.com/lmcinnes/umap

    Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.

- https://github.com/adamerose/PandasGUI

    A GUI for analyzing Pandas DataFrames.

- https://github.com/pandas-profiling/pandas-profiling

    Generates profile reports from a pandas DataFrame.

- https://github.com/facebookresearch/hiplot

    HiPlot is a lightweight interactive visualization tool to help AI researchers discover correlations and patterns in high-dimensional data using parallel plots and other graphical ways to represent information.

- https://github.com/DistrictDataLabs/yellowbrick

    Yellowbrick is a suite of visual diagnostic tools called "Visualizers" that extend the scikit-learn API to allow human steering of the model selection process. In a nutshell, Yellowbrick combines scikit-learn with matplotlib in the best tradition of the scikit-learn documentation, but to produce visualizations for your machine learning workflow!

- https://github.com/martinfleis/clustergram

    Clustergram - Visualization and diagnostics for cluster analysis in Python 

- https://github.com/lutzroeder/netron

     Visualizer for neural network, deep learning, and machine learning models.

### AutoML

- https://github.com/shankarpandala/lazypredict

    Lazy Predict helps build a lot of basic models without much code and helps understand which models works better without any parameter tuning.

- https://github.com/automl/auto-sklearn

    auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

- https://github.com/keras-team/autokeras

    AutoKeras: An AutoML system based on Keras. It is developed by DATA Lab at Texas A&M University. The goal of AutoKeras is to make machine learning accessible for everyone.

###  Model debugging

- https://github.com/neptune-ai/neptune-client

    Neptune is a lightweight experiment logging/tracking tool that helps you with your machine learning experiments. Neptune is suitable for indvidual, commercial and research projects.

- https://github.com/wandb/client

    Use W&B to organize and analyze machine learning experiments. It's framework-agnostic and lighter than TensorBoard. Each time you run a script instrumented with wandb, we save your hyperparameters and output metrics. Visualize models over the course of training, and compare versions of your models easily. We also automatically track the state of your code, system metrics, and configuration parameters.

- https://github.com/uber/manifold

    Manifold is a model-agnostic visual debugging tool for machine learning.

    Understanding ML model performance and behavior is a non-trivial process, given the intrisic opacity of ML algorithms. Performance summary statistics such as AUC, RMSE, and others are not instructive enough for identifying what went wrong with a model or how to improve it.

    As a visual analytics tool, Manifold allows ML practitioners to look beyond overall summary metrics to detect which subset of data a model is inaccurately predicting. Manifold also explains the potential cause of poor model performance by surfacing the feature distribution difference between better and worse-performing subsets of data.

- https://github.com/PerceptiLabs/PerceptiLabs

    PerceptiLabs is a dataflow driven, visual API for TensorFlow that enables data scientists to work more efficiently with machine learning models and to gain more insight into their models. It wraps low-level TensorFlow code to create visual components, which allows users to visualize the model architecture as the model is being built.

    This visual approach lowers the barrier of entry for beginners while providing researchers and advanced users with code-level access to their models.

- https://github.com/evidentlyai/evidently

    Evidently helps evaluate machine learning models during validation and monitor them in production. The tool generates interactive visual reports and JSON profiles from pandas `DataFrame` or `csv` files. You can use visual reports for ad hoc analysis, debugging and team sharing, and JSON profiles to integrate Evidently in prediction pipelines or with other visualization tools.

- https://github.com/TeamHG-Memex/eli5

    ELI5 is a Python package which helps to debug machine learning models and explain their predictions.

- https://github.com/slundberg/shap

    SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions (see papers for details and citations).

### DL fast research and low level controll

- https://github.com/microsoft/DeepSpeed

    10x Larger Models, 10x Faster Training, Minimal Code Change

    DeepSpeed delivers extreme-scale model training for everyone, from data scientists training on massive supercomputers to those training on low-end clusters or even on a single GPU:

    - Extreme scale: Using current generation of GPU clusters with hundreds of devices, 3D parallelism of DeepSpeed can efficiently train deep learning models with trillions of parameters.
    - Extremely memory efficient: With just a single GPU, ZeRO-Offload of DeepSpeed can train models with over 10B parameters, 10x bigger than the state of arts, democratizing multi-billion-parameter model training such that many deep learning scientists can explore bigger and better models.
    - Extremely long sequence length: Sparse attention of DeepSpeed powers an order-of-magnitude longer input sequence and obtains up to 6x faster execution comparing with dense transformers.
    - Extremely communication efficient: 3D parallelism improves communication efficiency allows users to train multi-billion-parameter models 2–7x faster on clusters with limited network bandwidth. 1-bit Adam/1-bit LAMB reduce communication volume by up to 5x while achieving similar convergence efficiency to Adam/LAMB, allowing for scaling to different types of GPU clusters and networks.

- https://github.com/apache/incubator-mxnet

    Apache MXNet is a deep learning framework designed for both efficiency and flexibility. It allows you to mix symbolic and imperative programming to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scalable to many GPUs and machines.

- https://github.com/PyTorchLightning/pytorch-lightning

    The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.

- https://github.com/catalyst-team/catalyst

    Accelerated deep learning R&D.

    PyTorch framework for Deep Learning research and development. It focuses on reproducibility, rapid experimentation, and codebase reuse so you can create something new rather than write another regular train loop.

    Supports fast `stochastic weight averaging, model tracing, model quantization, model pruning`.

- https://github.com/tensorflow/addons

    TensorFlow Addons is a repository of contributions that conform to well-established API patterns, but implement new functionality not available in core TensorFlow. TensorFlow natively supports a large number of operators, layers, metrics, losses, and optimizers. However, in a fast moving field like ML, there are many interesting new developments that cannot be integrated into core TensorFlow (because their broad applicability is not yet clear, or it is mostly used by a smaller subset of the community).

- https://github.com/tflearn/tflearn

    TFlearn is a modular and transparent deep learning library built on top of Tensorflow. It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.

### DL frameworks for fast prototyping and pretrained models

- https://github.com/huggingface/transformers

    Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages. Its aim is to make cutting-edge NLP easier to use for everyone.

    Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments.

- https://github.com/pytorch/fairseq

    Fairseq(-py) is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks.

    We provide reference implementations of various sequence modeling papers:

- https://github.com/NVIDIA/NeMo

    NVIDIA NeMo is a conversational AI toolkit built for researchers working on automatic speech recognition (ASR), natural language processing (NLP), and text-to-speech synthesis (TTS). The primary objective of NeMo is to help researchers from industry and academia to reuse prior work (code and pretrained models and make it easier to create new conversational AI models.

- https://github.com/speechbrain/speechbrain

    SpeechBrain is an open-source and all-in-one speech toolkit based on PyTorch.

    The goal is to create a single, flexible, and user-friendly toolkit that can be used to easily develop state-of-the-art speech technologies, including systems for speech recognition, speaker recognition, speech enhancement, multi-microphone signal processing and many others.

### NLP

- https://github.com/huggingface/tokenizers

    Provides an implementation of today's most used tokenizers, with a focus on performance and versatility.

- https://github.com/natasha/natasha

    Natasha solves basic NLP tasks for Russian language: tokenization, sentence segmentation, word embedding, morphology tagging, lemmatization, phrase normalization, syntax parsing, NER tagging, fact extraction. Quality on every task is similar or better then current SOTAs for Russian language on news articles, see evaluation section. Natasha is not a research project, underlying technologies are built for production. We pay attention to model size, RAM usage and performance. Models run on CPU, use Numpy for inference.

- https://github.com/allenai/allennlp

    AllenNLP is an open source library for building deep learning models for natural language processing, developed by the Allen Institute for Artificial Intelligence. It is built on top of PyTorch and is designed to support researchers, engineers, students, etc., who wish to build high quality deep NLP models with ease. It provides high-level abstractions and APIs for common components and models in modern NLP. It also provides an extensible framework that makes it easy to run and manage NLP experiments.

    In a nutshell, AllenNLP is:

    - a library with well-thought-out abstractions encapsulating the common data and model operations that are done in NLP research
    - a commandline tool for training PyTorch models
    - a collection of pre-trained models that you can use to make predictions
    - a collection of readable reference implementations of common / recent NLP models
    - an experiment framework for doing replicable science
    - a way to demo your research
    - open source and community driven

    AllenNLP is used by a large number of organizations and research projects.

- https://github.com/JohnSnowLabs/spark-nlp

    Spark NLP is a Natural Language Processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines that scale easily in a distributed environment. Spark NLP comes with 1100+ pretrained pipelines and models in more than 192+ languages. It supports state-of-the-art transformers such as BERT, XLNet, ELMO, ALBERT, and Universal Sentence Encoder that can be used seamlessly in a cluster. It also offers Tokenization, Word Segmentation, Part-of-Speech Tagging, Named Entity Recognition, Dependency Parsing, Spell Checking, Multi-class Text Classification, Multi-class Sentiment Analysis, Machine Translation (+180 languages), Summarization and Question Answering (Google T5), and many more NLP tasks.

- https://github.com/nlp-uoregon/trankit

    Trankit is a light-weight Transformer-based Python Toolkit for multilingual Natural Language Processing (NLP). It provides a trainable pipeline for fundamental NLP tasks over 100 languages, and 90 downloadable pretrained pipelines for 56 languages.

- https://github.com/nsu-ai/russian_g2p

    Accentor and transcriptor for Russian language 


### Audio processing


### CV

- https://github.com/NVlabs/imaginaire

    Imaginaire is a pytorch library that contains optimized implementation of several image and video synthesis methods developed at NVIDIA.

- https://github.com/rwightman/pytorch-image-models

    PyTorch image models, scripts, pretrained weights -- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more

- https://github.com/lukemelas/EfficientNet-PyTorch

    A PyTorch implementation of EfficientNet and EfficientNetV2 (coming soon!)

- https://github.com/dorarad/gansformer

    This is an implementation of the GANsformer model, a novel and efficient type of transformer, explored for the task of image generation. The network employs a bipartite structure that enables long-range interactions across the image, while maintaining computation of linearly efficiency, that can readily scale to high-resolution synthesis. The model iteratively propagates information from a set of latent variables to the evolving visual features and vice versa, to support the refinement of each in light of the other and encourage the emergence of compositional representations of objects and scenes. In contrast to the classic transformer architecture, it utilizes multiplicative integration that allows flexible region-based modulation, and can thus be seen as a generalization of the successful StyleGAN network.


### Time Series

- https://github.com/RJT1990/pyflux

    PyFlux is an open source time series library for Python. The library has a good array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models. By combining breadth of models with breadth of inference, PyFlux allows for a probabilistic approach to time series modelling.

- https://github.com/facebookresearch/Kats

    Kats is a toolkit to analyze time series data, a lightweight, easy-to-use, and generalizable framework to perform time series analysis. Time series analysis is an essential component of Data Science and Engineering work at industry, from understanding the key statistics and characteristics, detecting regressions and anomalies, to forecasting future trends. Kats aims to provide the one-stop shop for time series analysis, including detection, forecasting, feature extraction/embedding, multivariate analysis, etc.

- https://github.com/tslearn-team/tslearn

     The machine learning toolkit for time series analysis in Python.

- https://github.com/blue-yonder/tsfresh

     Automatic extraction of relevant features from time series.

- https://github.com/microsoft/qlib

    Qlib is an AI-oriented quantitative investment platform, which aims to realize the potential, empower the research, and create the value of AI technologies in quantitative investment.

    It contains the full ML pipeline of data processing, model training, back-testing; and covers the entire chain of quantitative investment: alpha seeking, risk modeling, portfolio optimization, and order execution.

    With Qlib, user can easily try ideas to create better Quant investment strategies.

- https://github.com/facebook/prophet

    Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

### Recommendations

- https://github.com/NicolasHug/Surprise

    Surprise is a Python scikit for building and analyzing recommender systems that deal with explicit rating data.

- https://github.com/tensorflow/recommenders

    TensorFlow Recommenders is a library for building recommender system models using TensorFlow.

    It helps with the full workflow of building a recommender system: data preparation, model formulation, training, evaluation, and deployment.

    It's built on Keras and aims to have a gentle learning curve while still giving you the flexibility to build complex models.

- https://github.com/microsoft/recommenders

    Best Practices on Recommendation Systems 

    This repository contains examples and best practices for building recommendation systems, provided as Jupyter notebooks.


### Distributed

- https://github.com/ray-project/ray

    Ray provides a simple, universal API for building distributed applications.

    Ray is packaged with the following libraries for accelerating machine learning workloads:

    - Tune: Scalable Hyperparameter Tuning
    - RLlib: Scalable Reinforcement Learning
    - RaySGD: Distributed Training Wrappers
    - Ray Serve: Scalable and Programmable Serving

- https://github.com/jmcarpenter2/swifter

    A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.

### Data mining and data processing

- https://github.com/clips/pattern

    Pattern is a web mining module for Python. It has tools for:

    - Data Mining: web services (Google, Twitter, Wikipedia), web crawler, HTML DOM parser
    - Natural Language Processing: part-of-speech taggers, n-gram search, sentiment analysis, WordNet
    - Machine Learning: vector space model, clustering, classification (KNN, SVM, Perceptron)
    - Network Analysis: graph centrality and visualization.

- https://github.com/SauceCat/pydqc

    Python automatic data quality check toolkit. Aims to relieve the pain of writing tedious codes for general data understanding by:

    - Automatically generate data summary report, which contains useful statistical information for each column in a data table. (useful for general data understanding)
    
    - Automatically summarize the statistical difference between two data tables. (useful for comparing training set with test set, comparing the same data table from two different snapshot dates, etc.)

- https://github.com/rafaelvalle/MDI

    This repository is associated with the paper "Missing Data Imputation for Supervised Learning" (arXiv), which empirically evaluates methods for imputing missing categorical data for supervised learning tasks.

- https://github.com/NathanEpstein/Dora

    Dora is a Python library designed to automate the painful parts of exploratory data analysis.

    The library contains convenience functions for data cleaning, feature selection & extraction, visualization, partitioning data for model validation, and versioning transformations of data.

    The library uses and is intended to be a helpful addition to common Python data analysis tools such as pandas, scikit-learn, and matplotlib.

- https://github.com/alirezamika/autoscraper

    This project is made for automatic web scraping to make scraping easy. It gets a url or the html content of a web page and a list of sample data which we want to scrape from that page. This data can be text, url or any html tag value of that page. It learns the scraping rules and returns the similar elements. Then you can use this learned object with new urls to get similar content or the exact same element of those new pages.

- https://github.com/pydata/pandas-datareader

    Extract data from a wide range of Internet sources into a pandas DataFrame.

    Functions from pandas_datareader.data and pandas_datareader.wb extract data from various Internet sources into a pandas DataFrame. Currently the following sources are supported: 

        Tiingo
        IEX
        Robinhood
        Alpha Vantage
        Enigma
        Quandl
        St.Louis FED (FRED)
        Kenneth French’s data library
        World Bank
        OECD
        Eurostat
        Thrift Savings Plan
        Nasdaq Trader symbol definitions
        Stooq
        MOEX


- https://github.com/google/textfsm

    Python module which implements a template based state machine for parsing semi-formatted text. Originally developed to allow programmatic access to information returned from the command line interface (CLI) of networking devices.

- https://github.com/avian2/unidecode

    It often happens that you have text data in Unicode, but you need to represent it in ASCII. For example when integrating with legacy code that doesn't support Unicode, or for ease of entry of non-Roman names on a US keyboard, or when constructing ASCII machine identifiers from human-readable Unicode strings that should still be somewhat intelligible. A popular example of this is when making an URL slug from an article title.

- https://github.com/deanmalmgren/textract

    Extract text from any document. no muss. no fuss.

- https://github.com/LuminosoInsight/python-ftfy

    Fixes mojibake and other glitches in Unicode text, after the fact.


### Enterprise ML and DL

- https://github.com/ikatsov/tensor-house

    TensorHouse is a collection of reference machine learning and optimization models for enterprise operations: marketing, pricing, supply chain, and more. The goal of the project is to provide baseline implementations for industrial, research, and educational purposes.

- https://github.com/quantumblacklabs/kedro

    Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code. It borrows concepts from software engineering and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning.

- https://github.com/Netflix/metaflow

    Metaflow is a human-friendly Python/R library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.

- https://github.com/apache/airflow

    Apache Airflow (or simply Airflow) is a platform to programmatically author, schedule, and monitor workflows. When workflows are defined as code, they become more maintainable, versionable, testable, and collaborative.

    Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies. Rich command line utilities make performing complex surgeries on DAGs a snap. The rich user interface makes it easy to visualize pipelines running in production, monitor progress, and troubleshoot issues when needed.

- https://github.com/dagster-io/dagster

    A data orchestrator for machine learning, analytics, and ETL. 

    Dagster lets you define pipelines in terms of the data flow between reusable, logical components, then test locally and run anywhere. With a unified view of pipelines and the assets they produce, Dagster can schedule and orchestrate Pandas, Spark, SQL, or anything else that Python can invoke.

    Dagster is designed for data platform engineers, data engineers, and full-stack data scientists. Building a data platform with Dagster makes your stakeholders more independent and your systems more robust. Developing data pipelines with Dagster makes testing easier and deploying faster.

### Speed up

- https://github.com/NVIDIA/DALI

    The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data. It can be used as a portable drop-in replacement for built in data loaders and data iterators in popular deep learning frameworks.

    DALI addresses the problem of the CPU bottleneck by offloading data preprocessing to the GPU. Additionally, DALI relies on its own execution engine, built to maximize the throughput of the input pipeline. Features such as prefetching, parallel execution, and batch processing are handled transparently for the user.

- https://github.com/modin-project/modin

    Scale your pandas workflows by changing one line of code

### Testing

- https://github.com/google/atheris
    
    Atheris is a coverage-guided Python fuzzing engine. It supports fuzzing of Python code, but also native extensions written for CPython. Atheris is based off of libFuzzer. When fuzzing native code, Atheris can be used in combination with Address Sanitizer or Undefined Behavior Sanitizer to catch extra bugs.

### Python app creator helpers:

- https://github.com/plotly/dash - web app

    Built on top of Plotly.js, React and Flask, Dash ties modern UI elements like dropdowns, sliders, and graphs directly to your analytical Python code. Read our tutorial proudly crafted ❤️ by Dash itself

- https://github.com/streamlit/streamlit

    Streamlit lets you turn data scripts into sharable web apps in minutes, not weeks. It's all Python, open-source, and free! And once you've created an app you can use our free sharing platform to deploy, manage, and share your app with the world.

- https://github.com/chriskiehl/Gooey

    Turn (almost) any Python 2 or 3 Console Program into a GUI application with one line

### Automate

- https://github.com/huginn/huginn

    Huginn is a system for building agents that perform automated tasks for you online. They can read the web, watch for events, and take actions on your behalf. Huginn's Agents create and consume events, propagating them along a directed graph. Think of it as a hackable version of IFTTT or Zapier on your own server. You always know who has your data. You do.


### G2P, word tokenizers
- https://github.com/AdolfVonKleist/Phonetisaurus

   This repository contains scripts suitable for training, evaluating and using grapheme-to-phoneme models for speech recognition using the OpenFst framework. The current build requires OpenFst version 1.6.0 or later, and the examples below use version 1.7.2.

- https://github.com/samzshi0529/HanziNLP?tab=readme-ov-file#1-quick-start

   Welcome to HanziNLP 🌟 - an ready-to-use toolkit for Natural Language Processing (NLP) on Chinese text, while also accommodating English. It is designed to be user-friendly and simplified tool even for freshmen in python like simple way for Chinese word tokenization and visualization via built-in Chinese fonts(Simplified Chinese).

- https://github.com/uhermjakob/utoken

   utoken is a multilingual tokenizer that divides text into words, punctuation and special tokens such as numbers, URLs, XML tags, email-addresses and hashtags. The tokenizer comes with a companion detokenizer. Initial public release of beta version 0.1.0 on Oct. 1, 2021.

- https://github.com/rhasspy/gruut

   A tokenizer, text cleaner, and IPA phonemizer for several human languages that supports SSML.

- https://github.com/dmort27/epitran

  A library and tool for transliterating orthographic text as IPA (International Phonetic Alphabet).


# coding: utf-8

# In[ ]:


# Plagiarism Project, Machine Learning Deployment

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker.

## Project Overview

In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

This project will be broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Load in the corpus of plagiarism text data.
* Explore the existing data features and the data distribution.
* This first notebook is **not** required in your final project submission.

**Notebook 2: Feature Engineering**

* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Select "good" features, by analyzing the correlations between different features.
* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Your Model in SageMaker**

* Upload your train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

---

Please see the [README](https://github.com/udacity/ML_SageMaker_Studies/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook and downloading the project files (as well as the other notebooks).


# In[ ]:


{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Plagiarism Text Data\n",
    "\n",
    "In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar the text file is when compared to a provided source text. \n",
    "\n",
    "The first step in working with any dataset is loading the data in and noting what information is included in the dataset. This is an important step in eventually working with this data, and knowing what kinds of features you have to work with as you transform and group the data!\n",
    "\n",
    "So, this notebook is all about exploring the data and noting patterns about the features you are given and the distribution of data. \n",
    "\n",
    "> There are not any exercises or questions in this notebook, it is only meant for exploration. This notebook will note be required in your final project submission.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Read in the Data\n",
    "\n",
    "The cell below will download the necessary data and extract the files into the folder `data/`.\n",
    "\n",
    "This data is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). \n",
    "\n",
    "> **Citation for data**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2019-07-08 23:48:31--  https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip\n",
      "Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.176.85\n",
      "Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.176.85|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 113826 (111K) [application/zip]\n",
      "Saving to: ‘data.zip’\n",
      "\n",
      "data.zip            100%[===================>] 111.16K  --.-KB/s    in 0.02s   \n",
      "\n",
      "2019-07-08 23:48:31 (4.83 MB/s) - ‘data.zip’ saved [113826/113826]\n",
      "\n",
      "Archive:  data.zip\n",
      "   creating: data/\n",
      "  inflating: data/.DS_Store          \n",
      "   creating: __MACOSX/\n",
      "   creating: __MACOSX/data/\n",
      "  inflating: __MACOSX/data/._.DS_Store  \n",
      "  inflating: data/file_information.csv  \n",
      "  inflating: __MACOSX/data/._file_information.csv  \n",
      "  inflating: data/g0pA_taska.txt     \n",
      "  inflating: __MACOSX/data/._g0pA_taska.txt  \n",
      "  inflating: data/g0pA_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g0pA_taskb.txt  \n",
      "  inflating: data/g0pA_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g0pA_taskc.txt  \n",
      "  inflating: data/g0pA_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g0pA_taskd.txt  \n",
      "  inflating: data/g0pA_taske.txt     \n",
      "  inflating: __MACOSX/data/._g0pA_taske.txt  \n",
      "  inflating: data/g0pB_taska.txt     \n",
      "  inflating: __MACOSX/data/._g0pB_taska.txt  \n",
      "  inflating: data/g0pB_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g0pB_taskb.txt  \n",
      "  inflating: data/g0pB_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g0pB_taskc.txt  \n",
      "  inflating: data/g0pB_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g0pB_taskd.txt  \n",
      "  inflating: data/g0pB_taske.txt     \n",
      "  inflating: __MACOSX/data/._g0pB_taske.txt  \n",
      "  inflating: data/g0pC_taska.txt     \n",
      "  inflating: __MACOSX/data/._g0pC_taska.txt  \n",
      "  inflating: data/g0pC_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g0pC_taskb.txt  \n",
      "  inflating: data/g0pC_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g0pC_taskc.txt  \n",
      "  inflating: data/g0pC_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g0pC_taskd.txt  \n",
      "  inflating: data/g0pC_taske.txt     \n",
      "  inflating: __MACOSX/data/._g0pC_taske.txt  \n",
      "  inflating: data/g0pD_taska.txt     \n",
      "  inflating: __MACOSX/data/._g0pD_taska.txt  \n",
      "  inflating: data/g0pD_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g0pD_taskb.txt  \n",
      "  inflating: data/g0pD_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g0pD_taskc.txt  \n",
      "  inflating: data/g0pD_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g0pD_taskd.txt  \n",
      "  inflating: data/g0pD_taske.txt     \n",
      "  inflating: __MACOSX/data/._g0pD_taske.txt  \n",
      "  inflating: data/g0pE_taska.txt     \n",
      "  inflating: __MACOSX/data/._g0pE_taska.txt  \n",
      "  inflating: data/g0pE_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g0pE_taskb.txt  \n",
      "  inflating: data/g0pE_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g0pE_taskc.txt  \n",
      "  inflating: data/g0pE_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g0pE_taskd.txt  \n",
      "  inflating: data/g0pE_taske.txt     \n",
      "  inflating: __MACOSX/data/._g0pE_taske.txt  \n",
      "  inflating: data/g1pA_taska.txt     \n",
      "  inflating: __MACOSX/data/._g1pA_taska.txt  \n",
      "  inflating: data/g1pA_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g1pA_taskb.txt  \n",
      "  inflating: data/g1pA_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g1pA_taskc.txt  \n",
      "  inflating: data/g1pA_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g1pA_taskd.txt  \n",
      "  inflating: data/g1pA_taske.txt     \n",
      "  inflating: __MACOSX/data/._g1pA_taske.txt  \n",
      "  inflating: data/g1pB_taska.txt     \n",
      "  inflating: __MACOSX/data/._g1pB_taska.txt  \n",
      "  inflating: data/g1pB_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g1pB_taskb.txt  \n",
      "  inflating: data/g1pB_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g1pB_taskc.txt  \n",
      "  inflating: data/g1pB_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g1pB_taskd.txt  \n",
      "  inflating: data/g1pB_taske.txt     \n",
      "  inflating: __MACOSX/data/._g1pB_taske.txt  \n",
      "  inflating: data/g1pD_taska.txt     \n",
      "  inflating: __MACOSX/data/._g1pD_taska.txt  \n",
      "  inflating: data/g1pD_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g1pD_taskb.txt  \n",
      "  inflating: data/g1pD_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g1pD_taskc.txt  \n",
      "  inflating: data/g1pD_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g1pD_taskd.txt  \n",
      "  inflating: data/g1pD_taske.txt     \n",
      "  inflating: __MACOSX/data/._g1pD_taske.txt  \n",
      "  inflating: data/g2pA_taska.txt     \n",
      "  inflating: __MACOSX/data/._g2pA_taska.txt  \n",
      "  inflating: data/g2pA_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g2pA_taskb.txt  \n",
      "  inflating: data/g2pA_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g2pA_taskc.txt  \n",
      "  inflating: data/g2pA_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g2pA_taskd.txt  \n",
      "  inflating: data/g2pA_taske.txt     \n",
      "  inflating: __MACOSX/data/._g2pA_taske.txt  \n",
      "  inflating: data/g2pB_taska.txt     \n",
      "  inflating: __MACOSX/data/._g2pB_taska.txt  \n",
      "  inflating: data/g2pB_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g2pB_taskb.txt  \n",
      "  inflating: data/g2pB_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g2pB_taskc.txt  \n",
      "  inflating: data/g2pB_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g2pB_taskd.txt  \n",
      "  inflating: data/g2pB_taske.txt     \n",
      "  inflating: __MACOSX/data/._g2pB_taske.txt  \n",
      "  inflating: data/g2pC_taska.txt     \n",
      "  inflating: __MACOSX/data/._g2pC_taska.txt  \n",
      "  inflating: data/g2pC_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g2pC_taskb.txt  \n",
      "  inflating: data/g2pC_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g2pC_taskc.txt  \n",
      "  inflating: data/g2pC_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g2pC_taskd.txt  \n",
      "  inflating: data/g2pC_taske.txt     \n",
      "  inflating: __MACOSX/data/._g2pC_taske.txt  \n",
      "  inflating: data/g2pE_taska.txt     \n",
      "  inflating: __MACOSX/data/._g2pE_taska.txt  \n",
      "  inflating: data/g2pE_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g2pE_taskb.txt  \n",
      "  inflating: data/g2pE_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g2pE_taskc.txt  \n",
      "  inflating: data/g2pE_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g2pE_taskd.txt  \n",
      "  inflating: data/g2pE_taske.txt     \n",
      "  inflating: __MACOSX/data/._g2pE_taske.txt  \n",
      "  inflating: data/g3pA_taska.txt     \n",
      "  inflating: __MACOSX/data/._g3pA_taska.txt  \n",
      "  inflating: data/g3pA_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g3pA_taskb.txt  \n",
      "  inflating: data/g3pA_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g3pA_taskc.txt  \n",
      "  inflating: data/g3pA_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g3pA_taskd.txt  \n",
      "  inflating: data/g3pA_taske.txt     \n",
      "  inflating: __MACOSX/data/._g3pA_taske.txt  \n",
      "  inflating: data/g3pB_taska.txt     \n",
      "  inflating: __MACOSX/data/._g3pB_taska.txt  \n",
      "  inflating: data/g3pB_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g3pB_taskb.txt  \n",
      "  inflating: data/g3pB_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g3pB_taskc.txt  \n",
      "  inflating: data/g3pB_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g3pB_taskd.txt  \n",
      "  inflating: data/g3pB_taske.txt     \n",
      "  inflating: __MACOSX/data/._g3pB_taske.txt  \n",
      "  inflating: data/g3pC_taska.txt     \n",
      "  inflating: __MACOSX/data/._g3pC_taska.txt  \n",
      "  inflating: data/g3pC_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g3pC_taskb.txt  \n",
      "  inflating: data/g3pC_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g3pC_taskc.txt  \n",
      "  inflating: data/g3pC_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g3pC_taskd.txt  \n",
      "  inflating: data/g3pC_taske.txt     \n",
      "  inflating: __MACOSX/data/._g3pC_taske.txt  \n",
      "  inflating: data/g4pB_taska.txt     \n",
      "  inflating: __MACOSX/data/._g4pB_taska.txt  \n",
      "  inflating: data/g4pB_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g4pB_taskb.txt  \n",
      "  inflating: data/g4pB_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g4pB_taskc.txt  \n",
      "  inflating: data/g4pB_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g4pB_taskd.txt  \n",
      "  inflating: data/g4pB_taske.txt     \n",
      "  inflating: __MACOSX/data/._g4pB_taske.txt  \n",
      "  inflating: data/g4pC_taska.txt     \n",
      "  inflating: __MACOSX/data/._g4pC_taska.txt  \n",
      "  inflating: data/g4pC_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g4pC_taskb.txt  \n",
      "  inflating: data/g4pC_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g4pC_taskc.txt  \n",
      "  inflating: data/g4pC_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g4pC_taskd.txt  \n",
      "  inflating: data/g4pC_taske.txt     \n",
      "  inflating: __MACOSX/data/._g4pC_taske.txt  \n",
      "  inflating: data/g4pD_taska.txt     \n",
      "  inflating: __MACOSX/data/._g4pD_taska.txt  \n",
      "  inflating: data/g4pD_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g4pD_taskb.txt  \n",
      "  inflating: data/g4pD_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g4pD_taskc.txt  \n",
      "  inflating: data/g4pD_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g4pD_taskd.txt  \n",
      "  inflating: data/g4pD_taske.txt     \n",
      "  inflating: __MACOSX/data/._g4pD_taske.txt  \n",
      "  inflating: data/g4pE_taska.txt     \n",
      "  inflating: __MACOSX/data/._g4pE_taska.txt  \n",
      "  inflating: data/g4pE_taskb.txt     \n",
      "  inflating: __MACOSX/data/._g4pE_taskb.txt  \n",
      "  inflating: data/g4pE_taskc.txt     \n",
      "  inflating: __MACOSX/data/._g4pE_taskc.txt  \n",
      "  inflating: data/g4pE_taskd.txt     \n",
      "  inflating: __MACOSX/data/._g4pE_taskd.txt  \n",
      "  inflating: data/g4pE_taske.txt     \n",
      "  inflating: __MACOSX/data/._g4pE_taske.txt  \n",
      "  inflating: data/orig_taska.txt     \n",
      "  inflating: __MACOSX/data/._orig_taska.txt  \n",
      "  inflating: data/orig_taskb.txt     \n",
      "  inflating: data/orig_taskc.txt     \n",
      "  inflating: __MACOSX/data/._orig_taskc.txt  \n",
      "  inflating: data/orig_taskd.txt     \n",
      "  inflating: __MACOSX/data/._orig_taskd.txt  \n",
      "  inflating: data/orig_taske.txt     \n",
      "  inflating: __MACOSX/data/._orig_taske.txt  \n",
      "  inflating: data/test_info.csv      \n",
      "  inflating: __MACOSX/data/._test_info.csv  \n",
      "  inflating: __MACOSX/._data         \n"
     ]
    }
   ],
   "source": [
    "!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip\n",
    "!unzip data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This plagiarism dataset is made of multiple text files; each of these files has characteristics that are is summarized in a `.csv` file named `file_information.csv`, which we can read in using `pandas`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>File</th>\n",
       "      <th>Task</th>\n",
       "      <th>Category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>g0pA_taska.txt</td>\n",
       "      <td>a</td>\n",
       "      <td>non</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>g0pA_taskb.txt</td>\n",
       "      <td>b</td>\n",
       "      <td>cut</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>g0pA_taskc.txt</td>\n",
       "      <td>c</td>\n",
       "      <td>light</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>g0pA_taskd.txt</td>\n",
       "      <td>d</td>\n",
       "      <td>heavy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>g0pA_taske.txt</td>\n",
       "      <td>e</td>\n",
       "      <td>non</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>g0pB_taska.txt</td>\n",
       "      <td>a</td>\n",
       "      <td>non</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>g0pB_taskb.txt</td>\n",
       "      <td>b</td>\n",
       "      <td>non</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>g0pB_taskc.txt</td>\n",
       "      <td>c</td>\n",
       "      <td>cut</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>g0pB_taskd.txt</td>\n",
       "      <td>d</td>\n",
       "      <td>light</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>g0pB_taske.txt</td>\n",
       "      <td>e</td>\n",
       "      <td>heavy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             File Task Category\n",
       "0  g0pA_taska.txt    a      non\n",
       "1  g0pA_taskb.txt    b      cut\n",
       "2  g0pA_taskc.txt    c    light\n",
       "3  g0pA_taskd.txt    d    heavy\n",
       "4  g0pA_taske.txt    e      non\n",
       "5  g0pB_taska.txt    a      non\n",
       "6  g0pB_taskb.txt    b      non\n",
       "7  g0pB_taskc.txt    c      cut\n",
       "8  g0pB_taskd.txt    d    light\n",
       "9  g0pB_taske.txt    e    heavy"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "csv_file = 'data/file_information.csv'\n",
    "plagiarism_df = pd.read_csv(csv_file)\n",
    "\n",
    "# print out the first few rows of data info\n",
    "plagiarism_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Types of Plagiarism\n",
    "\n",
    "Each text file is associated with one **Task** (task A-E) and one **Category** of plagiarism, which you can see in the above DataFrame.\n",
    "\n",
    "###  Five task types, A-E\n",
    "\n",
    "Each text file contains an answer to one short question; these questions are labeled as tasks A-E.\n",
    "* Each task, A-E, is about a topic that might be included in the Computer Science curriculum that was created by the authors of this dataset. \n",
    "    * For example, Task A asks the question: \"What is inheritance in object oriented programming?\"\n",
    "\n",
    "### Four categories of plagiarism \n",
    "\n",
    "Each text file has an associated plagiarism label/category:\n",
    "\n",
    "1. `cut`: An answer is plagiarized; it is copy-pasted directly from the relevant Wikipedia source text.\n",
    "2. `light`: An answer is plagiarized; it is based on the Wikipedia source text and includes some copying and paraphrasing.\n",
    "3. `heavy`: An answer is plagiarized; it is based on the Wikipedia source text but expressed using different words and structure. Since this doesn't copy directly from a source text, this will likely be the most challenging kind of plagiarism to detect.\n",
    "4. `non`: An answer is not plagiarized; the Wikipedia source text is not used to create this answer.\n",
    "5. `orig`: This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes.\n",
    "\n",
    "> So, out of the submitted files, the only category that does not contain any plagiarism is `non`.\n",
    "\n",
    "In the next cell, print out some statistics about the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of files:  100\n",
      "Number of unique tasks/question types (A-E):  5\n",
      "Unique plagiarism categories:  ['non' 'cut' 'light' 'heavy' 'orig']\n"
     ]
    }
   ],
   "source": [
    "# print out some stats about the data\n",
    "print('Number of files: ', plagiarism_df.shape[0])  # .shape[0] gives the rows \n",
    "# .unique() gives unique items in a specified column\n",
    "print('Number of unique tasks/question types (A-E): ', (len(plagiarism_df['Task'].unique())))\n",
    "print('Unique plagiarism categories: ', (plagiarism_df['Category'].unique()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You should see the number of text files in the dataset as well as some characteristics about the `Task` and `Category` columns. **Note that the file count of 100 *includes* the 5 _original_ wikipedia files for tasks A-E.** If you take a look at the files in the `data` directory, you'll notice that the original, source texts start with the filename `orig_` as opposed to `g` for \"group.\" \n",
    "\n",
    "> So, in total there are 100 files, 95 of which are answers (submitted by people) and 5 of which are the original, Wikipedia source texts.\n",
    "\n",
    "Your end goal will be to use this information to classify any given answer text into one of two categories, plagiarized or not-plagiarized."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Distribution of Data\n",
    "\n",
    "Next, let's look at the distribution of data. In this course, we've talked about traits like class imbalance that can inform how you develop an algorithm. So, here, we'll ask: **How evenly is our data distributed among different tasks and plagiarism levels?**\n",
    "\n",
    "Below, you should notice two things:\n",
    "* Our dataset is quite small, especially with respect to examples of varying plagiarism levels.\n",
    "* The data is distributed fairly evenly across task and plagiarism types."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Task:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Task</th>\n",
       "      <th>Counts</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>b</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>c</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>d</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>e</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Task  Counts\n",
       "0    a      20\n",
       "1    b      20\n",
       "2    c      20\n",
       "3    d      20\n",
       "4    e      20"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Plagiarism Levels:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Counts</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>cut</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>heavy</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>light</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>non</td>\n",
       "      <td>38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>orig</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category  Counts\n",
       "0      cut      19\n",
       "1    heavy      19\n",
       "2    light      19\n",
       "3      non      38\n",
       "4     orig       5"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Task & Plagiarism Level Combos :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Task</th>\n",
       "      <th>Category</th>\n",
       "      <th>Counts</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a</td>\n",
       "      <td>cut</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a</td>\n",
       "      <td>heavy</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a</td>\n",
       "      <td>light</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a</td>\n",
       "      <td>non</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a</td>\n",
       "      <td>orig</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>b</td>\n",
       "      <td>cut</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>b</td>\n",
       "      <td>heavy</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>b</td>\n",
       "      <td>light</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>b</td>\n",
       "      <td>non</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>b</td>\n",
       "      <td>orig</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>c</td>\n",
       "      <td>cut</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>c</td>\n",
       "      <td>heavy</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>c</td>\n",
       "      <td>light</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>c</td>\n",
       "      <td>non</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>c</td>\n",
       "      <td>orig</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>d</td>\n",
       "      <td>cut</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>d</td>\n",
       "      <td>heavy</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>d</td>\n",
       "      <td>light</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>d</td>\n",
       "      <td>non</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>d</td>\n",
       "      <td>orig</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>e</td>\n",
       "      <td>cut</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>e</td>\n",
       "      <td>heavy</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>e</td>\n",
       "      <td>light</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>e</td>\n",
       "      <td>non</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>e</td>\n",
       "      <td>orig</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Task Category  Counts\n",
       "0     a      cut       4\n",
       "1     a    heavy       3\n",
       "2     a    light       3\n",
       "3     a      non       9\n",
       "4     a     orig       1\n",
       "5     b      cut       3\n",
       "6     b    heavy       4\n",
       "7     b    light       3\n",
       "8     b      non       9\n",
       "9     b     orig       1\n",
       "10    c      cut       3\n",
       "11    c    heavy       5\n",
       "12    c    light       4\n",
       "13    c      non       7\n",
       "14    c     orig       1\n",
       "15    d      cut       4\n",
       "16    d    heavy       4\n",
       "17    d    light       5\n",
       "18    d      non       6\n",
       "19    d     orig       1\n",
       "20    e      cut       5\n",
       "21    e    heavy       3\n",
       "22    e    light       4\n",
       "23    e      non       7\n",
       "24    e     orig       1"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Show counts by different tasks and amounts of plagiarism\n",
    "\n",
    "# group and count by task\n",
    "counts_per_task=plagiarism_df.groupby(['Task']).size().reset_index(name=\"Counts\")\n",
    "print(\"\\nTask:\")\n",
    "display(counts_per_task)\n",
    "\n",
    "# group by plagiarism level\n",
    "counts_per_category=plagiarism_df.groupby(['Category']).size().reset_index(name=\"Counts\")\n",
    "print(\"\\nPlagiarism Levels:\")\n",
    "display(counts_per_category)\n",
    "\n",
    "# group by task AND plagiarism level\n",
    "counts_task_and_plagiarism=plagiarism_df.groupby(['Task', 'Category']).size().reset_index(name=\"Counts\")\n",
    "print(\"\\nTask & Plagiarism Level Combos :\")\n",
    "display(counts_task_and_plagiarism)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It may also be helpful to look at this last DataFrame, graphically.\n",
    "\n",
    "Below, you can see that the counts follow a pattern broken down by task. Each task has one source text (original) and the highest number on `non` plagiarized cases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 25 artists>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "% matplotlib inline\n",
    "\n",
    "# counts\n",
    "group = ['Task', 'Category']\n",
    "counts = plagiarism_df.groupby(group).size().reset_index(name=\"Counts\")\n",
    "\n",
    "plt.figure(figsize=(8,5))\n",
    "plt.bar(range(len(counts)), counts['Counts'], color = 'blue')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Up Next\n",
    "\n",
    "This notebook is just about data loading and exploration, and you do not need to include it in your final project submission. \n",
    "\n",
    "In the next few notebooks, you'll use this data to train a complete plagiarism classifier. You'll be tasked with extracting meaningful features from the text data, reading in answers to different tasks and comparing them to the original Wikipedia source text. You'll engineer similarity features that will help identify cases of plagiarism. Then, you'll use these features to train and deploy a classification model in a SageMaker notebook instance. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_amazonei_mxnet_p36",
   "language": "python",
   "name": "conda_amazonei_mxnet_p36"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}


# In[ ]:


{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Plagiarism Detection, Feature Engineering\n",
    "\n",
    "In this project, you will be tasked with building a plagiarism detector that examines an answer text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar that text file is to a provided, source text. \n",
    "\n",
    "Your first task will be to create some features that can then be used to train a classification model. This task will be broken down into a few discrete steps:\n",
    "\n",
    "* Clean and pre-process the data.\n",
    "* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.\n",
    "* Select \"good\" features, by analyzing the correlations between different features.\n",
    "* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.\n",
    "\n",
    "In the _next_ notebook, Notebook 3, you'll use the features and `.csv` files you create in _this_ notebook to train a binary classification model in a SageMaker notebook instance.\n",
    "\n",
    "You'll be defining a few different similarity features, as outlined in [this paper](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c412841_developing-a-corpus-of-plagiarised-short-answers/developing-a-corpus-of-plagiarised-short-answers.pdf), which should help you build a robust plagiarism detector!\n",
    "\n",
    "To complete this notebook, you'll have to complete all given exercises and answer all the questions in this notebook.\n",
    "> All your tasks will be clearly labeled **EXERCISE** and questions as **QUESTION**.\n",
    "\n",
    "It will be up to you to decide on the features to include in your final training and test data.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Read in the Data\n",
    "\n",
    "The cell below will download the necessary, project data and extract the files into the folder `data/`.\n",
    "\n",
    "This data is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). \n",
    "\n",
    "> **Citation for data**: Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [Download]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2019-07-10 23:17:34--  https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip\n",
      "Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.115.21\n",
      "Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.115.21|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 113826 (111K) [application/zip]\n",
      "Saving to: ‘data.zip.1’\n",
      "\n",
      "data.zip.1          100%[===================>] 111.16K  --.-KB/s    in 0.02s   \n",
      "\n",
      "2019-07-10 23:17:34 (4.90 MB/s) - ‘data.zip.1’ saved [113826/113826]\n",
      "\n",
      "Archive:  data.zip\n",
      "replace data/.DS_Store? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C\n"
     ]
    }
   ],
   "source": [
    "# NOTE:\n",
    "# you only need to run this cell if you have not yet downloaded the data\n",
    "# otherwise you may skip this cell or comment it out\n",
    "\n",
    "!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip\n",
    "!unzip data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This plagiarism dataset is made of multiple text files; each of these files has characteristics that are is summarized in a `.csv` file named `file_information.csv`, which we can read in using `pandas`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>File</th>\n",
       "      <th>Task</th>\n",
       "      <th>Category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>g0pA_taska.txt</td>\n",
       "      <td>a</td>\n",
       "      <td>non</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>g0pA_taskb.txt</td>\n",
       "      <td>b</td>\n",
       "      <td>cut</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>g0pA_taskc.txt</td>\n",
       "      <td>c</td>\n",
       "      <td>light</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>g0pA_taskd.txt</td>\n",
       "      <td>d</td>\n",
       "      <td>heavy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>g0pA_taske.txt</td>\n",
       "      <td>e</td>\n",
       "      <td>non</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             File Task Category\n",
       "0  g0pA_taska.txt    a      non\n",
       "1  g0pA_taskb.txt    b      cut\n",
       "2  g0pA_taskc.txt    c    light\n",
       "3  g0pA_taskd.txt    d    heavy\n",
       "4  g0pA_taske.txt    e      non"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "csv_file = 'data/file_information.csv'\n",
    "plagiarism_df = pd.read_csv(csv_file)\n",
    "\n",
    "# print out the first few rows of data info\n",
    "plagiarism_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Types of Plagiarism\n",
    "\n",
    "Each text file is associated with one **Task** (task A-E) and one **Category** of plagiarism, which you can see in the above DataFrame.\n",
    "\n",
    "###  Tasks, A-E\n",
    "\n",
    "Each text file contains an answer to one short question; these questions are labeled as tasks A-E. For example, Task A asks the question: \"What is inheritance in object oriented programming?\"\n",
    "\n",
    "### Categories of plagiarism \n",
    "\n",
    "Each text file has an associated plagiarism label/category:\n",
    "\n",
    "**1. Plagiarized categories: `cut`, `light`, and `heavy`.**\n",
    "* These categories represent different levels of plagiarized answer texts. `cut` answers copy directly from a source text, `light` answers are based on the source text but include some light rephrasing, and `heavy` answers are based on the source text, but *heavily* rephrased (and will likely be the most challenging kind of plagiarism to detect).\n",
    "     \n",
    "**2. Non-plagiarized category: `non`.** \n",
    "* `non` indicates that an answer is not plagiarized; the Wikipedia source text is not used to create this answer.\n",
    "    \n",
    "**3. Special, source text category: `orig`.**\n",
    "* This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Pre-Process the Data\n",
    "\n",
    "In the next few cells, you'll be tasked with creating a new DataFrame of desired information about all of the files in the `data/` directory. This will prepare the data for feature extraction and for training a binary, plagiarism classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EXERCISE: Convert categorical to numerical data\n",
    "\n",
    "You'll notice that the `Category` column in the data, contains string or categorical values, and to prepare these for feature extraction, we'll want to convert these into numerical values. Additionally, our goal is to create a binary classifier and so we'll need a binary class label that indicates whether an answer text is plagiarized (1) or not (0). Complete the below function `numerical_dataframe` that reads in a `file_information.csv` file by name, and returns a *new* DataFrame with a numerical `Category` column and a new `Class` column that labels each answer as plagiarized or not. \n",
    "\n",
    "Your function should return a new DataFrame with the following properties:\n",
    "\n",
    "* 4 columns: `File`, `Task`, `Category`, `Class`. The `File` and `Task` columns can remain unchanged from the original `.csv` file.\n",
    "* Convert all `Category` labels to numerical labels according to the following rules (a higher value indicates a higher degree of plagiarism):\n",
    "    * 0 = `non`\n",
    "    * 1 = `heavy`\n",
    "    * 2 = `light`\n",
    "    * 3 = `cut`\n",
    "    * -1 = `orig`, this is a special value that indicates an original file.\n",
    "* For the new `Class` column\n",
    "    * Any answer text that is not plagiarized (`non`) should have the class label `0`. \n",
    "    * Any plagiarized answer texts should have the class label `1`. \n",
    "    * And any `orig` texts will have a special label `-1`. \n",
    "\n",
    "### Expected output\n",
    "\n",
    "After running your function, you should get a DataFrame with rows that looks like the following: \n",
    "```\n",
    "\n",
    "        File\t     Task  Category  Class\n",
    "0\tg0pA_taska.txt\ta\t  0   \t0\n",
    "1\tg0pA_taskb.txt\tb\t  3   \t1\n",
    "2\tg0pA_taskc.txt\tc\t  2   \t1\n",
    "3\tg0pA_taskd.txt\td\t  1   \t1\n",
    "4\tg0pA_taske.txt\te\t  0\t   0\n",
    "...\n",
    "...\n",
    "99   orig_taske.txt    e     -1      -1\n",
    "\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Read in a csv file and return a transformed dataframe\n",
    "def numerical_dataframe(csv_file='data/file_information.csv'):\n",
    "    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.\n",
    "       This function does two things: \n",
    "       1) converts `Category` column values to numerical values \n",
    "       2) Adds a new, numerical `Class` label column.\n",
    "       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.\n",
    "       Source texts have a special label, -1.\n",
    "       :param csv_file: The directory for the file_information.csv file\n",
    "       :return: A dataframe with numerical categories and a new `Class` label column'''\n",
    "    \n",
    "    # read in csv\n",
    "    df = pd.read_csv(csv_file)\n",
    "    \n",
    "    # replace string category with numerical category\n",
    "    numerical_categories = {'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1}\n",
    "    \n",
    "    for key, val in numerical_categories.items():\n",
    "        df = df.replace(key, val)\n",
    "        \n",
    "        \n",
    "    # add class labels \n",
    "    df['Class'] = np.where(df['Category'] != 0, 1, 0) # label whether plagiarized (1) or not (0)\n",
    "    df['Class'] = np.where(df['Category'] == -1, -1, df['Class']) # -1 category is origin => class is -1 (meaningless)\n",
    "    \n",
    "    return df\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test cells\n",
    "\n",
    "Below are a couple of test cells. The first is an informal test where you can check that your code is working as expected by calling your function and printing out the returned result.\n",
    "\n",
    "The **second** cell below is a more rigorous test cell. The goal of a cell like this is to ensure that your code is working as expected, and to form any variables that might be used in _later_ tests/code, in this case, the data frame, `transformed_df`.\n",
    "\n",
    "> The cells in this notebook should be run in chronological order (the order they appear in the notebook). This is especially important for test cells.\n",
    "\n",
    "Often, later cells rely on the functions, imports, or variables defined in earlier cells. For example, some tests rely on previous tests to work.\n",
    "\n",
    "These tests do not test all cases, but they are a great way to check that you are on the right track!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>File</th>\n",
       "      <th>Task</th>\n",
       "      <th>Category</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>g0pA_taska.txt</td>\n",
       "      <td>a</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>g0pA_taskb.txt</td>\n",
       "      <td>b</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>g0pA_taskc.txt</td>\n",
       "      <td>c</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>g0pA_taskd.txt</td>\n",
       "      <td>d</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>g0pA_taske.txt</td>\n",
       "      <td>e</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>g0pB_taska.txt</td>\n",
       "      <td>a</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>g0pB_taskb.txt</td>\n",
       "      <td>b</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>g0pB_taskc.txt</td>\n",
       "      <td>c</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>g0pB_taskd.txt</td>\n",
       "      <td>d</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>g0pB_taske.txt</td>\n",
       "      <td>e</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             File Task  Category  Class\n",
       "0  g0pA_taska.txt    a         0      0\n",
       "1  g0pA_taskb.txt    b         3      1\n",
       "2  g0pA_taskc.txt    c         2      1\n",
       "3  g0pA_taskd.txt    d         1      1\n",
       "4  g0pA_taske.txt    e         0      0\n",
       "5  g0pB_taska.txt    a         0      0\n",
       "6  g0pB_taskb.txt    b         0      0\n",
       "7  g0pB_taskc.txt    c         3      1\n",
       "8  g0pB_taskd.txt    d         2      1\n",
       "9  g0pB_taske.txt    e         1      1"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# informal testing, print out the results of a called function\n",
    "# create new `transformed_df`\n",
    "transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')\n",
    "\n",
    "# check work\n",
    "# check that all categories of plagiarism have a class label = 1\n",
    "transformed_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tests Passed!\n",
      "\n",
      "Example data: \n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>File</th>\n",
       "      <th>Task</th>\n",
       "      <th>Category</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>g0pA_taska.txt</td>\n",
       "      <td>a</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>g0pA_taskb.txt</td>\n",
       "      <td>b</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>g0pA_taskc.txt</td>\n",
       "      <td>c</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>g0pA_taskd.txt</td>\n",
       "      <td>d</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>g0pA_taske.txt</td>\n",
       "      <td>e</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             File Task  Category  Class\n",
       "0  g0pA_taska.txt    a         0      0\n",
       "1  g0pA_taskb.txt    b         3      1\n",
       "2  g0pA_taskc.txt    c         2      1\n",
       "3  g0pA_taskd.txt    d         1      1\n",
       "4  g0pA_taske.txt    e         0      0"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# test cell that creates `transformed_df`, if tests are passed\n",
    "\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "\n",
    "# importing tests\n",
    "import problem_unittests as tests\n",
    "\n",
    "# test numerical_dataframe function\n",
    "tests.test_numerical_df(numerical_dataframe)\n",
    "\n",
    "# if above test is passed, create NEW `transformed_df`\n",
    "transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')\n",
    "\n",
    "# check work\n",
    "print('\\nExample data: ')\n",
    "transformed_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Processing & Splitting Data\n",
    "\n",
    "Recall that the goal of this project is to build a plagiarism classifier. At it's heart, this task is a comparison text; one that looks at a given answer and a source text, compares them and predicts whether an answer has plagiarized from the source. To effectively do this comparison, and train a classifier we'll need to do a few more things: pre-process all of our text data and prepare the text files (in this case, the 95 answer files and 5 original source files) to be easily compared, and split our data into a `train` and `test` set that can be used to train a classifier and evaluate it, respectively. \n",
    "\n",
    "To this end, you've been provided code that adds  additional information to your `transformed_df` from above. The next two cells need not be changed; they add two additional columns to the `transformed_df`:\n",
    "\n",
    "1. A `Text` column; this holds all the lowercase text for a `File`, with extraneous punctuation removed.\n",
    "2. A `Datatype` column; this is a string value `train`, `test`, or `orig` that labels a data point as part of our train or test set\n",
    "\n",
    "The details of how these additional columns are created can be found in the `helpers.py` file in the project directory. You're encouraged to read through that file to see exactly how text is processed and how data is split.\n",
    "\n",
    "Run the cells below to get a `complete_df` that has all the information you need to proceed with plagiarism detection and feature engineering."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "import helpers \n",
    "\n",
    "# create a text column \n",
    "text_df = helpers.create_text_column(transformed_df)\n",
    "text_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# after running the cell above\n",
    "# check out the processed text for a single file, by row index\n",
    "row_idx = 0 # feel free to change this index\n",
    "\n",
    "sample_text = text_df.iloc[0]['Text']\n",
    "\n",
    "print('Sample processed text:\\n\\n', sample_text)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Split data into training and test sets\n",
    "\n",
    "The next cell will add a `Datatype` column to a given DataFrame to indicate if the record is: \n",
    "* `train` - Training data, for model training.\n",
    "* `test` - Testing data, for model evaluation.\n",
    "* `orig` - The task's original answer from wikipedia.\n",
    "\n",
    "### Stratified sampling\n",
    "\n",
    "The given code uses a helper function which you can view in the `helpers.py` file in the main project directory. This implements [stratified random sampling](https://en.wikipedia.org/wiki/Stratified_sampling) to randomly split data by task & plagiarism amount. Stratified sampling ensures that we get training and test data that is fairly evenly distributed across task & plagiarism combinations. Approximately 26% of the data is held out for testing and 74% of the data is used for training.\n",
    "\n",
    "The function **train_test_dataframe** takes in a DataFrame that it assumes has `Task` and `Category` columns, and, returns a modified frame that indicates which `Datatype` (train, test, or orig) a file falls into. This sampling will change slightly based on a passed in *random_seed*. Due to a small sample size, this stratified random sampling will provide more stable results for a binary plagiarism classifier. Stability here is smaller *variance* in the accuracy of classifier, given a random seed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "random_seed = 1 # can change; set for reproducibility\n",
    "\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "import helpers\n",
    "\n",
    "# create new df with Datatype (train, test, orig) column\n",
    "# pass in `text_df` from above to create a complete dataframe, with all the information you need\n",
    "complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)\n",
    "\n",
    "# check results\n",
    "complete_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Determining Plagiarism\n",
    "\n",
    "Now that you've prepared this data and created a `complete_df` of information, including the text and class associated with each file, you can move on to the task of extracting similarity features that will be useful for plagiarism classification. \n",
    "\n",
    "> Note: The following code exercises, assume that the `complete_df` as it exists now, will **not** have its existing columns modified. \n",
    "\n",
    "The `complete_df` should always include the columns: `['File', 'Task', 'Category', 'Class', 'Text', 'Datatype']`. You can add additional columns, and you can create any new DataFrames you need by copying the parts of the `complete_df` as long as you do not modify the existing values, directly.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "# Similarity Features \n",
    "\n",
    "One of the ways we might go about detecting plagiarism, is by computing **similarity features** that measure how similar a given answer text is as compared to the original wikipedia source text (for a specific task, a-e). The similarity features you will use are informed by [this paper on plagiarism detection](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c412841_developing-a-corpus-of-plagiarised-short-answers/developing-a-corpus-of-plagiarised-short-answers.pdf). \n",
    "> In this paper, researchers created features called **containment** and **longest common subsequence**. \n",
    "\n",
    "Using these features as input, you will train a model to distinguish between plagiarized and not-plagiarized text files.\n",
    "\n",
    "## Feature Engineering\n",
    "\n",
    "Let's talk a bit more about the features we want to include in a plagiarism detection model and how to calculate such features. In the following explanations, I'll refer to a submitted text file as a **Student Answer Text (A)** and the original, wikipedia source file (that we want to compare that answer to) as the **Wikipedia Source Text (S)**.\n",
    "\n",
    "### Containment\n",
    "\n",
    "Your first task will be to create **containment features**. To understand containment, let's first revisit a definition of [n-grams](https://en.wikipedia.org/wiki/N-gram). An *n-gram* is a sequential word grouping. For example, in a line like \"bayes rule gives us a way to combine prior knowledge with new information,\" a 1-gram is just one word, like \"bayes.\" A 2-gram might be \"bayes rule\" and a 3-gram might be \"combine prior knowledge.\"\n",
    "\n",
    "> Containment is defined as the **intersection** of the n-gram word count of the Wikipedia Source Text (S) with the n-gram word count of the Student  Answer Text (S) *divided* by the n-gram word count of the Student Answer Text.\n",
    "\n",
    "$$ \\frac{\\sum{count(\\text{ngram}_{A}) \\cap count(\\text{ngram}_{S})}}{\\sum{count(\\text{ngram}_{A})}} $$\n",
    "\n",
    "If the two texts have no n-grams in common, the containment will be 0, but if _all_ their n-grams intersect then the containment will be 1. Intuitively, you can see how having longer n-gram's in common, might be an indication of cut-and-paste plagiarism. In this project, it will be up to you to decide on the appropriate `n` or several `n`'s to use in your final model.\n",
    "\n",
    "### EXERCISE: Create containment features\n",
    "\n",
    "Given the `complete_df` that you've created, you should have all the information you need to compare any Student  Answer Text (A) with its appropriate Wikipedia Source Text (S). An answer for task A should be compared to the source text for task A, just as answers to tasks B, C, D, and E should be compared to the corresponding original source text.\n",
    "\n",
    "In this exercise, you'll complete the function, `calculate_containment` which calculates containment based upon the following parameters:\n",
    "* A given DataFrame, `df` (which is assumed to be the `complete_df` from above)\n",
    "* An `answer_filename`, such as 'g0pB_taskd.txt' \n",
    "* An n-gram length, `n`\n",
    "\n",
    "### Containment calculation\n",
    "\n",
    "The general steps to complete this function are as follows:\n",
    "1. From *all* of the text files in a given `df`, create an array of n-gram counts; it is suggested that you use a [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for this purpose.\n",
    "2. Get the processed answer and source texts for the given `answer_filename`.\n",
    "3. Calculate the containment between an answer and source text according to the following equation.\n",
    "\n",
    "    >$$ \\frac{\\sum{count(\\text{ngram}_{A}) \\cap count(\\text{ngram}_{S})}}{\\sum{count(\\text{ngram}_{A})}} $$\n",
    "    \n",
    "4. Return that containment value.\n",
    "\n",
    "You are encouraged to write any helper functions that you need to complete the function below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Calculate the ngram containment for one answer file/source file pair in a df\n",
    "def calculate_containment(df, n, answer_filename):\n",
    "    '''Calculates the containment between a given answer text and its associated source text.\n",
    "       This function creates a count of ngrams (of a size, n) for each text file in our data.\n",
    "       Then calculates the containment by finding the ngram count for a given answer text, \n",
    "       and its associated source text, and calculating the normalized intersection of those counts.\n",
    "       :param df: A dataframe with columns,\n",
    "           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'\n",
    "       :param n: An integer that defines the ngram size\n",
    "       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'\n",
    "       :return: A single containment value that represents the similarity\n",
    "           between an answer text and its source text.\n",
    "    '''\n",
    "    \n",
    "    # your code here\n",
    "    \n",
    "    pass\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test cells\n",
    "\n",
    "After you've implemented the containment function, you can test out its behavior. \n",
    "\n",
    "The cell below iterates through the first few files, and calculates the original category _and_ containment values for a specified n and file.\n",
    "\n",
    ">If you've implemented this correctly, you should see that the non-plagiarized have low or close to 0 containment values and that plagiarized examples have higher containment values, closer to 1.\n",
    "\n",
    "Note what happens when you change the value of n. I recommend applying your code to multiple files and comparing the resultant containment values. You should see that the highest containment values correspond to files with the highest category (`cut`) of plagiarism level."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# select a value for n\n",
    "n = 3\n",
    "\n",
    "# indices for first few files\n",
    "test_indices = range(5)\n",
    "\n",
    "# iterate through files and calculate containment\n",
    "category_vals = []\n",
    "containment_vals = []\n",
    "for i in test_indices:\n",
    "    # get level of plagiarism for a given file index\n",
    "    category_vals.append(complete_df.loc[i, 'Category'])\n",
    "    # calculate containment for given file and n\n",
    "    filename = complete_df.loc[i, 'File']\n",
    "    c = calculate_containment(complete_df, n, filename)\n",
    "    containment_vals.append(c)\n",
    "\n",
    "# print out result, does it make sense?\n",
    "print('Original category values: \\n', category_vals)\n",
    "print()\n",
    "print(str(n)+'-gram containment values: \\n', containment_vals)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# run this test cell\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "# test containment calculation\n",
    "# params: complete_df from before, and containment function\n",
    "tests.test_containment(complete_df, calculate_containment)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### QUESTION 1: Why can we calculate containment features across *all* data (training & test), prior to splitting the DataFrame for modeling? That is, what about the containment calculation means that the test and training data do not influence each other?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Longest Common Subsequence\n",
    "\n",
    "Containment a good way to find overlap in word usage between two documents; it may help identify cases of cut-and-paste as well as paraphrased levels of plagiarism. Since plagiarism is a fairly complex task with varying levels, it's often useful to include other measures of similarity. The paper also discusses a feature called **longest common subsequence**.\n",
    "\n",
    "> The longest common subsequence is the longest string of words (or letters) that are *the same* between the Wikipedia Source Text (S) and the Student Answer Text (A). This value is also normalized by dividing by the total number of words (or letters) in the  Student Answer Text. \n",
    "\n",
    "In this exercise, we'll ask you to calculate the longest common subsequence of words between two texts.\n",
    "\n",
    "### EXERCISE: Calculate the longest common subsequence\n",
    "\n",
    "Complete the function `lcs_norm_word`; this should calculate the *longest common subsequence* of words between a Student Answer Text and corresponding Wikipedia Source Text. \n",
    "\n",
    "It may be helpful to think of this in a concrete example. A Longest Common Subsequence (LCS) problem may look as follows:\n",
    "* Given two texts: text A (answer text) of length n, and string S (original source text) of length m. Our goal is to produce their longest common subsequence of words: the longest sequence of words that appear left-to-right in both texts (though the words don't have to be in continuous order).\n",
    "* Consider:\n",
    "    * A = \"i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents\"\n",
    "    * S = \"pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents\"\n",
    "\n",
    "* In this case, we can see that the start of each sentence of fairly similar, having overlap in the sequence of words, \"pagerank is a link analysis algorithm used by\" before diverging slightly. Then we **continue moving left -to-right along both texts** until we see the next common sequence; in this case it is only one word, \"google\". Next we find \"that\" and \"a\" and finally the same ending \"to each element of a hyperlinked set of documents\".\n",
    "* Below, is a clear visual of how these sequences were found, sequentially, in each text.\n",
    "\n",
    "<img src='notebook_ims/common_subseq_words.png' width=40% />\n",
    "\n",
    "* Now, those words appear in left-to-right order in each document, sequentially, and even though there are some words in between, we count this as the longest common subsequence between the two texts. \n",
    "* If I count up each word that I found in common I get the value 20. **So, LCS has length 20**. \n",
    "* Next, to normalize this value, divide by the total length of the student answer; in this example that length is only 27. **So, the function `lcs_norm_word` should return the value `20/27` or about `0.7408`.**\n",
    "\n",
    "In this way, LCS is a great indicator of cut-and-paste plagiarism or if someone has referenced the same source text multiple times in an answer."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### LCS, dynamic programming\n",
    "\n",
    "If you read through the scenario above, you can see that this algorithm depends on looking at two texts and comparing them word by word. You can solve this problem in multiple ways. First, it may be useful to `.split()` each text into lists of comma separated words to compare. Then, you can iterate through each word in the texts and compare them, adding to your value for LCS as you go. \n",
    "\n",
    "The method I recommend for implementing an efficient LCS algorithm is: using a matrix and dynamic programming. **Dynamic programming** is all about breaking a larger problem into a smaller set of subproblems, and building up a complete result without having to repeat any subproblems. \n",
    "\n",
    "This approach assumes that you can split up a large LCS task into a combination of smaller LCS tasks. Let's look at a simple example that compares letters:\n",
    "\n",
    "* A = \"ABCD\"\n",
    "* S = \"BD\"\n",
    "\n",
    "We can see right away that the longest subsequence of _letters_ here is 2 (B and D are in sequence in both strings). And we can calculate this by looking at relationships between each letter in the two strings, A and S.\n",
    "\n",
    "Here, I have a matrix with the letters of A on top and the letters of S on the left side:\n",
    "\n",
    "<img src='notebook_ims/matrix_1.png' width=40% />\n",
    "\n",
    "This starts out as a matrix that has as many columns and rows as letters in the strings S and O **+1** additional row and column, filled with zeros on the top and left sides. So, in this case, instead of a 2x4 matrix it is a 3x5.\n",
    "\n",
    "Now, we can fill this matrix up by breaking it into smaller LCS problems. For example, let's first look at the shortest substrings: the starting letter of A and S. We'll first ask, what is the Longest Common Subsequence between these two letters \"A\" and \"B\"? \n",
    "\n",
    "**Here, the answer is zero and we fill in the corresponding grid cell with that value.**\n",
    "\n",
    "<img src='notebook_ims/matrix_2.png' width=30% />\n",
    "\n",
    "Then, we ask the next question, what is the LCS between \"AB\" and \"B\"?\n",
    "\n",
    "**Here, we have a match, and can fill in the appropriate value 1**.\n",
    "\n",
    "<img src='notebook_ims/matrix_3_match.png' width=25% />\n",
    "\n",
    "If we continue, we get to a final matrix that looks as follows, with a **2** in the bottom right corner.\n",
    "\n",
    "<img src='notebook_ims/matrix_6_complete.png' width=25% />\n",
    "\n",
    "The final LCS will be that value **2** *normalized* by the number of n-grams in A. So, our normalized value is 2/4 = **0.5**.\n",
    "\n",
    "### The matrix rules\n",
    "\n",
    "One thing to notice here is that, you can efficiently fill up this matrix one cell at a time. Each grid cell only depends on the values in the grid cells that are directly on top and to the left of it, or on the diagonal/top-left. The rules are as follows:\n",
    "* Start with a matrix that has one extra row and column of zeros.\n",
    "* As you traverse your string:\n",
    "    * If there is a match, fill that grid cell with the value to the top-left of that cell *plus* one. So, in our case, when we found a matching B-B, we added +1 to the value in the top-left of the matching cell, 0.\n",
    "    * If there is not a match, take the *maximum* value from either directly to the left or the top cell, and carry that value over to the non-match cell.\n",
    "\n",
    "<img src='notebook_ims/matrix_rules.png' width=50% />\n",
    "\n",
    "After completely filling the matrix, **the bottom-right cell will hold the non-normalized LCS value**.\n",
    "\n",
    "This matrix treatment can be applied to a set of words instead of letters. Your function should apply this to the words in two texts and return the normalized LCS value."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Compute the normalized LCS given an answer text and a source text\n",
    "def lcs_norm_word(answer_text, source_text):\n",
    "    '''Computes the longest common subsequence of words in two texts; returns a normalized value.\n",
    "       :param answer_text: The pre-processed text for an answer text\n",
    "       :param source_text: The pre-processed text for an answer's associated source text\n",
    "       :return: A normalized LCS value'''\n",
    "    \n",
    "    # your code here\n",
    "    \n",
    "    pass\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test cells\n",
    "\n",
    "Let's start by testing out your code on the example given in the initial description.\n",
    "\n",
    "In the below cell, we have specified strings A (answer text) and S (original source text). We know that these texts have 20 words in common and the submitted answer is 27 words long, so the normalized, longest common subsequence should be 20/27.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Run the test scenario from above\n",
    "# does your function return the expected value?\n",
    "\n",
    "A = \"i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents\"\n",
    "S = \"pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents\"\n",
    "\n",
    "# calculate LCS\n",
    "lcs = lcs_norm_word(A, S)\n",
    "print('LCS = ', lcs)\n",
    "\n",
    "\n",
    "# expected value test\n",
    "assert lcs==20/27., \"Incorrect LCS value, expected about 0.7408, got \"+str(lcs)\n",
    "\n",
    "print('Test passed!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This next cell runs a more rigorous test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# run test cell\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "# test lcs implementation\n",
    "# params: complete_df from before, and lcs_norm_word function\n",
    "tests.test_lcs(complete_df, lcs_norm_word)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, take a look at a few resultant values for `lcs_norm_word`. Just like before, you should see that higher values correspond to higher levels of plagiarism."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# test on your own\n",
    "test_indices = range(5) # look at first few files\n",
    "\n",
    "category_vals = []\n",
    "lcs_norm_vals = []\n",
    "# iterate through first few docs and calculate LCS\n",
    "for i in test_indices:\n",
    "    category_vals.append(complete_df.loc[i, 'Category'])\n",
    "    # get texts to compare\n",
    "    answer_text = complete_df.loc[i, 'Text'] \n",
    "    task = complete_df.loc[i, 'Task']\n",
    "    # we know that source texts have Class = -1\n",
    "    orig_rows = complete_df[(complete_df['Class'] == -1)]\n",
    "    orig_row = orig_rows[(orig_rows['Task'] == task)]\n",
    "    source_text = orig_row['Text'].values[0]\n",
    "    \n",
    "    # calculate lcs\n",
    "    lcs_val = lcs_norm_word(answer_text, source_text)\n",
    "    lcs_norm_vals.append(lcs_val)\n",
    "\n",
    "# print out result, does it make sense?\n",
    "print('Original category values: \\n', category_vals)\n",
    "print()\n",
    "print('Normalized LCS values: \\n', lcs_norm_vals)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Create All Features\n",
    "\n",
    "Now that you've completed the feature calculation functions, it's time to actually create multiple features and decide on which ones to use in your final model! In the below cells, you're provided two helper functions to help you create multiple features and store those in a DataFrame, `features_df`.\n",
    "\n",
    "### Creating multiple containment features\n",
    "\n",
    "Your completed `calculate_containment` function will be called in the next cell, which defines the helper function `create_containment_features`. \n",
    "\n",
    "> This function returns a list of containment features, calculated for a given `n` and for *all* files in a df (assumed to the the `complete_df`).\n",
    "\n",
    "For our original files, the containment value is set to a special value, -1.\n",
    "\n",
    "This function gives you the ability to easily create several containment features, of different n-gram lengths, for each of our text files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "# Function returns a list of containment features, calculated for a given n \n",
    "# Should return a list of length 100 for all files in a complete_df\n",
    "def create_containment_features(df, n, column_name=None):\n",
    "    \n",
    "    containment_values = []\n",
    "    \n",
    "    if(column_name==None):\n",
    "        column_name = 'c_'+str(n) # c_1, c_2, .. c_n\n",
    "    \n",
    "    # iterates through dataframe rows\n",
    "    for i in df.index:\n",
    "        file = df.loc[i, 'File']\n",
    "        # Computes features using calculate_containment function\n",
    "        if df.loc[i,'Category'] > -1:\n",
    "            c = calculate_containment(df, n, file)\n",
    "            containment_values.append(c)\n",
    "        # Sets value to -1 for original tasks \n",
    "        else:\n",
    "            containment_values.append(-1)\n",
    "    \n",
    "    print(str(n)+'-gram containment features created!')\n",
    "    return containment_values\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating LCS features\n",
    "\n",
    "Below, your complete `lcs_norm_word` function is used to create a list of LCS features for all the answer files in a given DataFrame (again, this assumes you are passing in the `complete_df`. It assigns a special value for our original, source files, -1.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "# Function creates lcs feature and add it to the dataframe\n",
    "def create_lcs_features(df, column_name='lcs_word'):\n",
    "    \n",
    "    lcs_values = []\n",
    "    \n",
    "    # iterate through files in dataframe\n",
    "    for i in df.index:\n",
    "        # Computes LCS_norm words feature using function above for answer tasks\n",
    "        if df.loc[i,'Category'] > -1:\n",
    "            # get texts to compare\n",
    "            answer_text = df.loc[i, 'Text'] \n",
    "            task = df.loc[i, 'Task']\n",
    "            # we know that source texts have Class = -1\n",
    "            orig_rows = df[(df['Class'] == -1)]\n",
    "            orig_row = orig_rows[(orig_rows['Task'] == task)]\n",
    "            source_text = orig_row['Text'].values[0]\n",
    "\n",
    "            # calculate lcs\n",
    "            lcs = lcs_norm_word(answer_text, source_text)\n",
    "            lcs_values.append(lcs)\n",
    "        # Sets to -1 for original tasks \n",
    "        else:\n",
    "            lcs_values.append(-1)\n",
    "\n",
    "    print('LCS features created!')\n",
    "    return lcs_values\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EXERCISE: Create a features DataFrame by selecting an `ngram_range`\n",
    "\n",
    "The paper suggests calculating the following features: containment *1-gram to 5-gram* and *longest common subsequence*. \n",
    "> In this exercise, you can choose to create even more features, for example from *1-gram to 7-gram* containment features and *longest common subsequence*. \n",
    "\n",
    "You'll want to create at least 6 features to choose from as you think about which to give to your final, classification model. Defining and comparing at least 6 different features allows you to discard any features that seem redundant, and choose to use the best features for your final model!\n",
    "\n",
    "In the below cell **define an n-gram range**; these will be the n's you use to create n-gram containment features. The rest of the feature creation code is provided."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Define an ngram range\n",
    "ngram_range = range(1,7)\n",
    "\n",
    "\n",
    "# The following code may take a minute to run, depending on your ngram_range\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "features_list = []\n",
    "\n",
    "# Create features in a features_df\n",
    "all_features = np.zeros((len(ngram_range)+1, len(complete_df)))\n",
    "\n",
    "# Calculate features for containment for ngrams in range\n",
    "i=0\n",
    "for n in ngram_range:\n",
    "    column_name = 'c_'+str(n)\n",
    "    features_list.append(column_name)\n",
    "    # create containment features\n",
    "    all_features[i]=np.squeeze(create_containment_features(complete_df, n))\n",
    "    i+=1\n",
    "\n",
    "# Calculate features for LCS_Norm Words \n",
    "features_list.append('lcs_word')\n",
    "all_features[i]= np.squeeze(create_lcs_features(complete_df))\n",
    "\n",
    "# create a features dataframe\n",
    "features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)\n",
    "\n",
    "# Print all features/columns\n",
    "print()\n",
    "print('Features: ', features_list)\n",
    "print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# print some results \n",
    "features_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Correlated Features\n",
    "\n",
    "You should use feature correlation across the *entire* dataset to determine which features are ***too*** **highly-correlated** with each other to include both features in a single model. For this analysis, you can use the *entire* dataset due to the small sample size we have. \n",
    "\n",
    "All of our features try to measure the similarity between two texts. Since our features are designed to measure similarity, it is expected that these features will be highly-correlated. Many classification models, for example a Naive Bayes classifier, rely on the assumption that features are *not* highly correlated; highly-correlated features may over-inflate the importance of a single feature. \n",
    "\n",
    "So, you'll want to choose your features based on which pairings have the lowest correlation. These correlation values range between 0 and 1; from low to high correlation, and are displayed in a [correlation matrix](https://www.displayr.com/what-is-a-correlation-matrix/), below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "# Create correlation matrix for just Features to determine different models to test\n",
    "corr_matrix = features_df.corr().abs().round(2)\n",
    "\n",
    "# display shows all of a dataframe\n",
    "display(corr_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EXERCISE: Create selected train/test data\n",
    "\n",
    "Complete the `train_test_data` function below. This function should take in the following parameters:\n",
    "* `complete_df`: A DataFrame that contains all of our processed text data, file info, datatypes, and class labels\n",
    "* `features_df`: A DataFrame of all calculated features, such as containment for ngrams, n= 1-5, and lcs values for each text file listed in the `complete_df` (this was created in the above cells)\n",
    "* `selected_features`: A list of feature column names,  ex. `['c_1', 'lcs_word']`, which will be used to select the final features in creating train/test sets of data.\n",
    "\n",
    "It should return two tuples:\n",
    "* `(train_x, train_y)`, selected training features and their corresponding class labels (0/1)\n",
    "* `(test_x, test_y)`, selected training features and their corresponding class labels (0/1)\n",
    "\n",
    "** Note: x and y should be arrays of feature values and numerical class labels, respectively; not DataFrames.**\n",
    "\n",
    "Looking at the above correlation matrix, you should decide on a **cutoff** correlation value, less than 1.0, to determine which sets of features are *too* highly-correlated to be included in the final training and test data. If you cannot find features that are less correlated than some cutoff value, it is suggested that you increase the number of features (longer n-grams) to choose from or use *only one or two* features in your final model to avoid introducing highly-correlated features.\n",
    "\n",
    "Recall that the `complete_df` has a `Datatype` column that indicates whether data should be `train` or `test` data; this should help you split the data appropriately."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Takes in dataframes and a list of selected features (column names) \n",
    "# and returns (train_x, train_y), (test_x, test_y)\n",
    "def train_test_data(complete_df, features_df, selected_features):\n",
    "    '''Gets selected training and test features from given dataframes, and \n",
    "       returns tuples for training and test features and their corresponding class labels.\n",
    "       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels\n",
    "       :param features_df: A dataframe of all computed, similarity features\n",
    "       :param selected_features: An array of selected features that correspond to certain columns in `features_df`\n",
    "       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''\n",
    "    \n",
    "    # get the training features\n",
    "    train_x = None\n",
    "    # And training class labels (0 or 1)\n",
    "    train_y = None\n",
    "    \n",
    "    # get the test features and labels\n",
    "    test_x = None\n",
    "    test_y = None\n",
    "    \n",
    "    return (train_x, train_y), (test_x, test_y)\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test cells\n",
    "\n",
    "Below, test out your implementation and create the final train/test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "test_selection = list(features_df)[:2] # first couple columns as a test\n",
    "# test that the correct train/test data is created\n",
    "(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, test_selection)\n",
    "\n",
    "# params: generated train/test data\n",
    "tests.test_data_split(train_x, train_y, test_x, test_y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EXERCISE: Select \"good\" features\n",
    "\n",
    "If you passed the test above, you can create your own train/test data, below. \n",
    "\n",
    "Define a list of features you'd like to include in your final mode, `selected_features`; this is a list of the features names you want to include."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Select your list of features, this should be column names from features_df\n",
    "# ex. ['c_1', 'lcs_word']\n",
    "selected_features = ['c_1', 'c_5', 'lcs_word']\n",
    "\n",
    "\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "\n",
    "(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, selected_features)\n",
    "\n",
    "# check that division of samples seems correct\n",
    "# these should add up to 95 (100 - 5 original files)\n",
    "print('Training size: ', len(train_x))\n",
    "print('Test size: ', len(test_x))\n",
    "print()\n",
    "print('Training df sample: \\n', train_x[:10])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 2: How did you decide on which features to include in your final model? "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Answer:**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## Creating Final Data Files\n",
    "\n",
    "Now, you are almost ready to move on to training a model in SageMaker!\n",
    "\n",
    "You'll want to access your train and test data in SageMaker and upload it to S3. In this project, SageMaker will expect the following format for your train/test data:\n",
    "* Training and test data should be saved in one `.csv` file each, ex `train.csv` and `test.csv`\n",
    "* These files should have class  labels in the first column and features in the rest of the columns\n",
    "\n",
    "This format follows the practice, outlined in the [SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html), which reads: \"Amazon SageMaker requires that a CSV file doesn't have a header record and that the target variable [class label] is in the first column.\"\n",
    "\n",
    "## EXERCISE: Create csv files\n",
    "\n",
    "Define a function that takes in x (features) and y (labels) and saves them to one `.csv` file at the path `data_dir/filename`.\n",
    "\n",
    "It may be useful to use pandas to merge your features and labels into one DataFrame and then convert that into a csv file. You can make sure to get rid of any incomplete rows, in a DataFrame, by using `dropna`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def make_csv(x, y, filename, data_dir):\n",
    "    '''Merges features and labels and converts them into one csv file with labels in the first column.\n",
    "       :param x: Data features\n",
    "       :param y: Data labels\n",
    "       :param file_name: Name of csv file, ex. 'train.csv'\n",
    "       :param data_dir: The directory where files will be saved\n",
    "       '''\n",
    "    # make data dir, if it does not exist\n",
    "    if not os.path.exists(data_dir):\n",
    "        os.makedirs(data_dir)\n",
    "    \n",
    "    \n",
    "    # your code here\n",
    "    \n",
    "    \n",
    "    # nothing is returned, but a print statement indicates that the function has run\n",
    "    print('Path created: '+str(data_dir)+'/'+str(filename))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test cells\n",
    "\n",
    "Test that your code produces the correct format for a `.csv` file, given some text features and labels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "fake_x = [ [0.39814815, 0.0001, 0.19178082], \n",
    "           [0.86936937, 0.44954128, 0.84649123], \n",
    "           [0.44086022, 0., 0.22395833] ]\n",
    "\n",
    "fake_y = [0, 1, 1]\n",
    "\n",
    "make_csv(fake_x, fake_y, filename='to_delete.csv', data_dir='test_csv')\n",
    "\n",
    "# read in and test dimensions\n",
    "fake_df = pd.read_csv('test_csv/to_delete.csv', header=None)\n",
    "\n",
    "# check shape\n",
    "assert fake_df.shape==(3, 4), \\\n",
    "      'The file should have as many rows as data_points and as many columns as features+1 (for indices).'\n",
    "# check that first column = labels\n",
    "assert np.all(fake_df.iloc[:,0].values==fake_y), 'First column is not equal to the labels, fake_y.'\n",
    "print('Tests passed!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# delete the test csv file, generated above\n",
    "! rm -rf test_csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If you've passed the tests above, run the following cell to create `train.csv` and `test.csv` files in a directory that you specify! This will save the data in a local directory. Remember the name of this directory because you will reference it again when uploading this data to S3."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# can change directory, if you want\n",
    "data_dir = 'plagiarism_data'\n",
    "\n",
    "\"\"\"\n",
    "DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE\n",
    "\"\"\"\n",
    "\n",
    "make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)\n",
    "make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Up Next\n",
    "\n",
    "Now that you've done some feature engineering and created some training and test data, you are ready to train and deploy a plagiarism classification model. The next notebook will utilize SageMaker resources to train and test a model that you design."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_amazonei_mxnet_p36",
   "language": "python",
   "name": "conda_amazonei_mxnet_p36"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}


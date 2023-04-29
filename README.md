# Facify

### modern. end-to-end facial verification system made with PyTorch, FastAPI, and Docker

The motivation behind this project is to explore how to build a robust end-to-end system from start to finish for efficient and fast facial verification. The goal build a API that can take in 2 photos and determine with extremely high confidence whether or not the same individual is in the photo. 

### Ideation

Occam’s razor is the problem solving principle that recommends searching for explanation constructed with the smallest possible set of elements. In other words, easy solution fix big problem. Building each component from the ground up allows us to build up this solution with Lego bricks. Each element inherently builds upon a level of abstractions which allows us to use whichever suite of technologies we want. Here is the rough idea:

1. client sends a request to the API to verify 2 photos
2. API takes the 2 photos and performs some preprocessing steps on it
3. API runs the photos through a model to get it’s embeddings
4. API computes the distance between the 2 embeddings
5. API returns wether the distance was below or above the similarity threshold

So we aim to build 1) a robust machine learning model and 2) a high-level serviceable API to build our end-to-end facial verification system. We'll be using PyTorch to train our models, FastAPI to write our API and Docker to package our builds.

## Machine Learning Model

We first start off my building up our machine learning model. Our machine learning model is used to process inputs and convert them into a feature-rich low-dimension space. We can than compare these low-dimensional embeddings to ascertain similarity between images.

![embedding](https://www.pinecone.io/images/vector_embeddings.jpg)
[Source: Pinecone](https://www.pinecone.io/learn/vector-embeddings/)

There is an entire suite of academic literature dedicated to building, training, and validating the most efficient and accurate facial recognition models. Models like VGG-Face, Google FaceNet, OpenFace are just a few that have pushed forwards the frontiers of facial recognition.

### Ideation

![pipeline](https://www.researchgate.net/publication/344375713/figure/fig2/AS:939581314723840@1601025050284/An-overview-of-the-proposed-face-recognition-pipeline.png)
[Source:](https://www.researchgate.net/figure/An-overview-of-the-proposed-face-recognition-pipeline_fig2_344375713)

We want to begin with

## Forward Serving API
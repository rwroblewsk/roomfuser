# RoomFuser: Room Impulse Response Generation using Neural Diffusion Models

## Authors

Rebecca Wroblewski, CCRMA
Eric Grinstein and Zehua Chen, Imperial College London

## Introduction

This project explores the usage of Denoising Diffusion Probabilistic Models (DDPMs) for the task
of acoustic Room Impulse Response (RIR) generation.

This is a fork of the original Roomfuser project by Eric Grinstein and Zehua Chen for a subsequent investigation by Rebecca Wroblewski

A RIR is defined for two points in a room, namely, a source and receiver (i.e., a microphone) position. It encodes
the three propagation effects which are applied to the raw or "dry" source signal, delay, attenuation and
reverberation. Examples of rooms with a high reverberation time are cathedrals and bathrooms, both of which
are good for singing. However, high reverberation also reduces speech inteligibility, which is a problem for
hard of hearing people.

RIRs are frequently generated using the [Image Source Method (ISM)](https://pubs.aip.org/asa/jasa/article/65/4/943/765693/Image-method-for-efficiently-simulating-small-room). However, this method is known to be computationally expensive.
The [FAST-RIR](https://github.com/anton-jeran/FAST-RIR) project proposes to train a Generative Adversarial Network (GAN) to 
learn RIRs, and shows that they are indeed faster than ISM.

As Diffusion Models have been shown to frequently [outclass GANs](https://openreview.net/pdf?id=AAWuCvzaVt) in terms of generation quality,

## Model details

This fork of the roomfuser project focused on using the [DiffWave](https://arxiv.org/abs/2009.09761) model architecture on the time domain RIR (https://github.com/lucidrains/denoising-diffusion-pytorch).

### Conditioning

We condition the DiffWave model on a vector of size 10, namely, the room dimensions, source position, receiver position and reverberation time (RT60).
The conditioning is applied in a similar way as in the original DiffWave paper, by adding it as a bias after the convolutions.

## Datasets
A combination of the original data from FAST-RIR as well as newly generated data (generated using pygsound) was used. 

## Usage
There are three main scripts:
* `python visualize_backward.py`, which generates RIRs from the conditioning.
* `python visualize_forward.py`, which transforms RIRs into noise, i.e., the forward process.
* `python -m roomfuser logs`, which trains the model
* `python evaluate_model.py`, runs the evaluation script for the model.

The generate_rirs.py script can also be used to generate more data, similarly to that generated for FAST-RIR.

### Configuration
You can play with different feature extractors, etc in the file `params.json` file

## Citing

If you use this in your research, please cite it. You can get a BibTeX citation by clicking "Cite this repository" on the right corner of this project's homepage. 
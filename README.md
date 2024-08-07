# SARA: Singular-Value Based Adaptive Low-Rank Adaption

## Overview

This repository contains implementations of SARA (Singular-Value Based Adaptive Low-Rank Adaption) and Mo-SARA (Mixture-of-SARA), novel parameter-efficient fine-tuning methods for large language models.

## Code Explanation

The code provides two main classes:

1. `SARA`: Implements the basic SARA method.
   - Adaptively determines the rank `k` based on singular values.
   - Adds a trainable low-rank matrix to the frozen pretrained weights.

2. `MoSARA`: Implements the Mixture-of-SARA method.
   - Extends SARA with multiple sets of trainable singular values.
   - Uses a routing mechanism to combine outputs from different experts.

## Paper Summary

The paper "SARA: Singular-Value Based Adaptive Low-Rank Adaption" introduces two methods for efficient fine-tuning of large language models:

### Key Contributions:

1. Analysis of the relationship between SVD results of pretrained model parameters and performance across layers.
2. SARA: A method to adaptively find appropriate ranks for each layer during initialization.
3. Mo-SARA: An extension that further reduces trainable parameters by using a mixture of experts approach.

### Main Findings:

- Different layers in transformers have varying degrees of importance.
- The number of singular values accounting for a certain proportion of the total sum (k) correlates with layer performance.
- SARA and Mo-SARA achieve comparable or better performance than existing methods while using fewer parameters.

### Advantages:

- Adaptive rank selection for each layer
- Improved parameter efficiency
- Retention of LoRA's benefits (no inference overhead)
- Applicability to various tasks and models

### Experimental Results:

The methods were tested on 15 datasets covering mathematical reasoning, commonsense inference, and E2E tasks, showing consistent improvements over baseline methods.

## Usage

To use these methods, initialize the SARA or Mo-SARA layers with pretrained weights and integrate them into your model architecture. Fine-tune as you would with other PEFT methods.

## Conclusion

SARA and Mo-SARA offer promising approaches for efficient fine-tuning of large language models, potentially advancing the field of parameter-efficient fine-tuning (PEFT) methods.
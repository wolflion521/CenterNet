from models.py_utils._cpools import top_pool, bottom_pool,left_pool,right_pool
import torch
import numpy as np
import matplotlib.pyplot as plt
import mypool

def test_1():
    # test what is the input and output of top_pool.cpp
    input = torch.randn(1,2,10,10)
    print("input",input)
    output = right_pool.forward(input)[0]
    print("output",output)
    print("output shape: ", output.shape)
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(input[0][0])
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(output[0][0])
    plt.show()

def test_2():
    # test what is the input and output of top_pool.cpp
    input = torch.randn(1,1, 5, 5)
    print("input", input)
    output = mypool.forward(input)[0]
    print("output", output)
    print("output shape: ", output.shape)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(input[0][0])
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(output[0][0])
    plt.show()

if __name__ == "__main__":
    test_1()
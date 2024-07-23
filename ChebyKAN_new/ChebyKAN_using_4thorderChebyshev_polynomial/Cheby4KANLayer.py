import torch
import torch.nn as nn

# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class Cheby4KANLayer(nn.Module):                   #define a custom neural network layer using Pytorch's nn.Module base #class
    def __init__(self, input_dim, output_dim, degree):  #constructor method for initializing the layer
        super(Cheby4KANLayer, self).__init__()       #call the constructor of the nn.Module base class
        self.inputdim = input_dim               #store the input dimension of the layer
        self.outdim = output_dim               #store the output dimension of the layer
        self.degree = degree                  #store the degree of the Chebyshev polynomial to be used
        self.cheby4_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))#initializes the Chebyshev coefficients as trainable parameters
        nn.init.normal_(self.cheby4_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))#initializes the coefficients from a normal distribution with                                                                 #mean 0 and standard deviation based on the input and degree
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))            #create a buffer named 'arange' containing a tensor with values                                                               #'[0,1,...,degree]'. Buffers are not updated during backprop
    def forward(self, x): #define the forward pass of the layer
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )                                           # shape = (batch_size, inputdim, self.degree + 1)
                                                   #reshapes and repeats 'x' to have shape (batch_size, inputdim, self.degree + 1)
                                           #apply inverse cosine function to 'x'
        # Multiply by arange [0 .. degree]
        x *= (2*self.arange + 1)/2                                #multiplies 'x' elementwise by the buffer 'arange'
        # Apply cos
        x = x.sin()                                          #apply sine function to 'x'
        #apply division
        x = x / sin(x / 2)
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby4_coeffs
        )                                            # shape = (batch_size, outdim)
        #compute Chebyshev interpolation using Einstein summation notation. This operation performs a dot product over specified dimensions.
        y = y.view(-1, self.outdim)                          #reshapes 'y' to have shape '(batch_size, outdim)'
        return y

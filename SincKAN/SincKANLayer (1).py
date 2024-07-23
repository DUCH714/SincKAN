import torch
import torch.nn as nn

class SincKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(SincKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.Sinc_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, 5 * (degree + 1)))  # Adjusted shape for 10 tensors

        nn.init.normal_(self.Sinc_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        N1 = torch.floor(torch.tensor(self.degree / 2)).int()
        N2 = torch.ceil(torch.tensor(self.degree / 2)).int()

        batch_size = x.size(0)
        x = x.view((batch_size, self.inputdim, 1)).expand(batch_size, self.inputdim, self.degree + 1)

        k = torch.arange(-N1, N2 + 1, device=x.device).view(1, 1, self.degree + 1)
        k = k.expand(batch_size, self.inputdim, self.degree + 1)

        tensors = []

        for i in range(1, 21, 4):  # Range from 1 to 20 with step of 2
            h = 1 / i
            tensor = x / h + k
            tensors.append(tensor)

        # Concatenate along the third dimension (dim=2)
        result = torch.cat(tensors, dim=2)

        # Using sin(x_j + k) / (x_j + k) for interpolation
        x_interp = torch.sin(torch.pi * (result + 1e-20)) / (torch.pi * (result + 1e-20))

        # Compute the interpolation using sin(x_j + k) / (x_j + k)
        y = torch.einsum("bid,iod->bo", x_interp, self.Sinc_coeffs)
        y = y.view(batch_size, self.outdim)
        return y


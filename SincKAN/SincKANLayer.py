import torch
import torch.nn as nn

class SincKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(SincKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        
        # Initialize h as a trainable parameter
        self.h = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(degree + 1)])
        
        self.Sinc_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, 5 * (degree + 1)))  # Adjusted shape for 5 times the degree + 1 tensors
        nn.init.normal_(self.Sinc_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        N1 = torch.floor(torch.tensor(self.degree / 2)).int().item()
        N2 = torch.ceil(torch.tensor(self.degree / 2)).int().item()

        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim, 1).expand(batch_size, self.input_dim, self.degree + 1)
        
        # Create five different x_adjusted tensors
        x_adjusted_list = []
        k = torch.arange(-N1, N2 + 1, device=x.device).view(1, 1, self.degree + 1).expand(batch_size, self.input_dim, self.degree + 1)
        
        for _ in range(5):
            x_adjusted = torch.stack([x[:, :, i] / self.h[i] for i in range(self.degree + 1)], dim=2)
            x_adjusted = x_adjusted + k  # Adjust x with k
            x_adjusted_list.append(x_adjusted)
        
        # Concatenate them along dim=2
        x_adjusted_concat = torch.cat(x_adjusted_list, dim=2)
        
        # Using sin(x_j + k) / (x_j + k) for interpolation
        x_interp = torch.sin(torch.pi * (x_adjusted_concat + 1e-20)) / (torch.pi * (x_adjusted_concat + 1e-20))
    
        # Compute the interpolation using sin(x_j + k) / (x_j + k)
        y = torch.einsum("bid,iod->bo", x_interp, self.Sinc_coeffs)
        y = y.view(batch_size, self.output_dim)
        return y




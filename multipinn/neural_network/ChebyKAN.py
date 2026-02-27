import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1) * output_dim**0.5))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        x = torch.tanh(x)
        x = torch.clamp(x, -0.99999, 0.99999)
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )
        y = y.view(-1, self.outdim)
        return y
    

class ChebyKAN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 layers_hidden,
                 spline_order=3,
                 ):
        super(ChebyKAN, self).__init__()
        self.spline_order = spline_order
        layers_hidden.insert(0, input_dim)
        layers_hidden.append(output_dim)

        self.layers = torch.nn.ModuleList()
        for index, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layers.append(
                ChebyKANLayer(
                    in_features,
                    out_features,
                    self.spline_order
                )
            )
            if index < len(layers_hidden) - 2:
                self.layers.append(
                    nn.LayerNorm(
                        out_features
                    )
                )

    
    def forward(self, x: torch.Tensor):
        for index, layer in enumerate(self.layers):
            x = layer(x)
        return(x)
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvLSTM, self).__init__()
        self._W_i = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self._U_i = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self._W_f = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self._U_f = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self._W_c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self._U_c = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self._W_o = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self._U_o = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, (h, c)):
        x_i = self._W_i(x)
        x_f = self._W_f(x)
        x_c = self._W_c(x)
        x_o = self._W_o(x)

        i = F.sigmoid(x_i + self._U_i(h))
        f = F.sigmoid(x_f + self._U_f(h))
        c = f*c + i*F.tanh(x_c + self._U_c(h))
        o = F.sigmoid(x_o + self._U_o(h))

        h = o * F.tanh(c)
        return h, c

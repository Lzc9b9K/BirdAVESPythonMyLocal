import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class FiveLayerNN(nn.Module):    
    def __init__(self, input_size=1024, hidden_size=256, output_size=4, dropout_rate=0.5, leaky_relu_slope=0.01):
        super(FiveLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=self.leaky_relu_slope)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leaky_relu_slope)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leaky_relu_slope)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x), negative_slope=self.leaky_relu_slope)
        x = self.dropout(x)
        # y = self.fc5(x) 
        y = F.softmax(self.fc5(x), dim=1)
        return y

class ThreeLayerNN(nn.Module):    
    def __init__(self, input_size=1024, hidden_size=256, output_size=4, dropout_rate=0.5, leaky_relu_slope=0.01):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = F.silu(self.fc2(x))
        x = self.dropout(x)
        # print(self.fc3(x))
        # print(self.fc3(x).shape)
        # y = self.fc3(x)
        y = F.softmax(self.fc3(x), dim=1)
        # print(y)
        return y

class OneLayerNN(nn.Module):    
    def __init__(self, input_size=1024, output_size=4):
        super(OneLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = F.softmax(self.fc1(x), dim=1)
        # print(y)
        return y

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Parameter für CNN definieren

filter1_size = 128
filter2_size = 32
kernel_size = 2
stride = 1
pool_size = 2

#Testdaten laden und ausgeben
test_input = pd.read_csv('Daten_5_Tage_Hauptstromzahler_Schule.csv')
#print('Test Array', np.loadtxt('Daten_5_Tage_Hauptstromzahler_Schule.csv', delimiter=';'))
#test_input = torch.randn(2,10)
print('Test Input:', test_input)
print(test_input.shape)
print('iloc', test_input)




#CNN Modell bauen
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        ## Schichten des CNN definieren

        self.conv1 = nn.Conv1d(2,filter1_size,kernel_size,stride,padding=0)

        self.conv2 = nn.Conv1d(filter1_size, filter2_size, kernel_size, stride, padding=0)

        self.maxpool = nn.MaxPool1d(pool_size)


        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        return x



cnn = CNN()
print('Netz:')
print(cnn)

#Netz laden, wenn vorhanden
if os.path.isfile('cnn.pt'):
    cnn = torch.load('cnn.pt')
    print('CNN geladen')

#mit Ziel- und Ausgangsdaten durch Netz iterieren
for i in range(5):

    testout = cnn(test_input)
    #Netz trainieren und evaluieren

    ziel_out = testout*2
    criterion = nn.MSELoss()
    loss = criterion(testout, ziel_out)
    print('Loss:', loss)

    cnn.zero_grad()
    loss.backward()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    optimizer.step()
#Netz speichern
torch.save(cnn, 'cnn.pt')
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime as dt
import time
import locale

#Testdaten laden und ausgeben
test_input = pd.read_csv('Daten_5_Tage_Hauptstromzahler_Schule.csv', sep=';')
#test_input = torch.randn(2,100)
print('Test Input:', test_input)
print(test_input.shape)
#print(test_input['Verbrauch'][1:5])

#Testdaten vorbereiten:
#Verbräuche als Floats speichern, Kommata entfernen
print('Zeilenanzahl Testdaten:', len(test_input.index))
for i in range(0, len(test_input.index)):
    test_input['Verbrauch'][i] = test_input['Verbrauch'][i].replace(',','.')
   # print(test_input['Verbrauch'][i])

#Datenspalte als Daten speichern
for i in range (0, len(test_input.index)):
    test_input['Datum'][i] = dt.datetime.strptime(test_input['Datum'][i], '%d.%m.%Y, %H:%M')
#Überführung in Unix Zeitstempel
for i in range(0, len(test_input.index)):
    test_input['Datum'][i] = time.mktime(test_input['Datum'][i].timetuple())
print('Testdaten als Unix Zeitstempel gespeichert', test_input['Datum'])

#Testdaten in richtigem Format für CNN speichern:
#Unix Zeitstempel als Integer speichern
test_input['Datum'] = torch.tensor(test_input['Datum'].values.astype(np.int32))
print('Daten in test_input als Int gespeichert', test_input)
print('Datentyp der Werte in Spalte Datum:', type(test_input['Datum'][1]))
#Verbrauch als Floats speichern
test_input['Verbrauch']= torch.tensor(test_input['Verbrauch'].values.astype(np.float32))
print('Verbrauch als floats gespeichert', test_input)
print('Datentyp der Werte in Spalte Verbrauch:', type(test_input['Verbrauch'][1]))

#Testdaten als Tensor speichern
test_input = torch.tensor(test_input.values)
print('Tensor Shape danach:', test_input.shape)
print('Typ danach:', type(test_input))
print('Tensor aus Testdaten:', test_input)
#Zeilen und Spalten des Tensors tauschen für CNN, damit 2 Channels vorliegen
test_input = torch.transpose(test_input, 0, 1)
print('Tensor aus Testdaten nach Zeilen- und Spaltentausch:', test_input)

#Parameter für CNN definieren

filter1_size = 128
filter2_size = 32
kernel_size = 2
stride = 1
pool_size = 2



#CNN Modell erstellen
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        ## Schichten des CNN definieren

        self.conv1 = nn.Conv1d(1,filter1_size,kernel_size,stride,padding=0)

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

#Referenztensor erstellen:
#test_tensor = torch.randn(2, 10)
#print(test_tensor)
#print('Shape des Test Tensors:', test_tensor.shape)
#print('Typ des Test Tensors:', type(test_input))

#mit Ziel- und Ausgangsdaten durch Netz iterieren
for i in range(5):

    testout = cnn(test_input)
    #Netz trainieren und evaluieren

    ziel_out = test_input*2
    criterion = nn.MSELoss()
    loss = criterion(test_input, ziel_out)
    print('Loss:', loss)

    cnn.zero_grad()
    loss.backward()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    optimizer.step()
#Netz speichern
torch.save(cnn, 'cnn.pt')
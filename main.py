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
#print('Tensor Shape danach:', test_input.shape)
#print('Typ danach:', type(test_input))
#print('Tensor aus Testdaten:', test_input)

#Zeilen und Spalten des Tensors tauschen für CNN, damit 2 Channels vorliegen
#test_input = torch.transpose(test_input, 0, 1)

#print('Tensor aus Testdaten nach Zeilen- und Spaltentausch:', test_input)
#Tensor für CNN von float64 in float32 ändern
test_input = test_input.float()
#print(test_input)

#Aufteilung der Daten in Trainings-, Test- und Validierungsdaten:
#Bestimmung der Anteile und jeweiligen Daten
#print('Länge Test Input:', test_input.shape)
train_part = int(0.3*len(test_input))
#print('Werte Training:', train_part)
valid_part = int(0.1*len(test_input))
#print('Werte Validierung:', valid_part)
test_part = int(0.6*len(test_input))
#print('Werte Test:', test_part)

train_data = list(test_input[:train_part,0])
valid_data = list(test_input[train_part:train_part+valid_part,0])
test_data = list(test_input[train_part+valid_part:,0])


#Parameter für CNN definieren

win = 30    #Größe der betrachteten historischen Sequenz (history window)
pred = 1    #Größe der vorhergesagten Sequenz (prediction window)
filter1_size = 128
filter2_size = 32
kernel_size = 2
stride = 1
pool_size = 2

#Sequenzen für das Training bestimmen

def get_subsequences(data):
    X = []
    Y = []

    for i in range(len(data) - win - pred):
        X.append(data[i:i+win])
        Y.append(data[i+win:i+win+pred])
    return np.array(X),np.array(Y)

trainX,trainY = get_subsequences(train_data)
trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))

validX,validY = get_subsequences(valid_data)
validX = np.reshape(validX,(validX.shape[0],1,validX.shape[1]))

testX,testY = get_subsequences(test_data)
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

#CNN Modell erstellen
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        ## Schichten des CNN definieren

        self.conv1 = nn.Conv1d(1,filter1_size,kernel_size,stride,padding=0)

        self.conv2 = nn.Conv1d(filter1_size, filter2_size, kernel_size, stride, padding=0)

        self.maxpool = nn.MaxPool1d(pool_size)

        self.dim1 = int(0.5*(0.5*(win-1)-1)) * filter2_size

        self.lin1 = nn.Linear(self.dim1, len(test_input))

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        #1. Convolutional Layer und Maxpool
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        #2. Convolutional Layer und Maxpool
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        #
        x = x.view(-1, self.dim1)

        x = self.dropout(x)

        x = self.lin1(x)
        return x


# CNN Modell definieren
cnn = CNN()
print('Netz:', cnn)

# Optimizer definieren
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.10)

#Funktion zum Modelltraining

def train(epochs,trainX,trainY,validX,validY,model,optimizer,criterion,save_path,freq = 5):

    target_train = torch.tensor(trainY).type('torch.FloatTensor')
    data_train = torch.tensor(trainX).type('torch.FloatTensor')

    target_valid = torch.tensor(validY).type('torch.FloatTensor')
    data_valid = torch.tensor(validX).type('torch.FloatTensor')

    train_loss_min = np.Inf
    valid_loss_min = np.Inf
    last_valid_loss = 0

    for epoch in range (1, epochs + 1):
        #Training:
        model.train()

        optimizer.zero_grad()
        output = model(data_train)
        loss = criterion(output, target_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        #Validierung:
        model.eval()
        output_valid = model(data_valid)

        loss_valid = criterion(output_valid, target_valid)
        valid_loss = loss_valid.item()
        if(valid_loss == last_valid_loss):
            print('problem')

        last_valid_loss = valid_loss
        if(epoch%freq == 0):
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

        if valid_loss < valid_loss_min:
            print('Validierungsverlust niedriger ({:.6f} --> {:.6f}). Speichere Modell...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    return model, output


#Modell trainieren
cnn,out = train(10, trainX,trainY,validX,validY, cnn, optimizer, criterion, 'cnn.pt', freq = 1)




#Netz laden, wenn vorhanden
#if os.path.isfile('cnn.pt'):
#    cnn = torch.load('cnn.pt')
#    print('CNN geladen')

#Referenztensor erstellen:
#test_tensor = torch.randn(2, 10)
#print(test_tensor)
#print('Shape des Test Tensors:', test_tensor.shape)
#print('Typ des Test Tensors:', type(test_input))

#Zieloutput des Netzes definieren (muss gleiche Dimensionen haben wie Netzoutput)
#ziel_out = torch.randn(32,126)


#mit Ziel- und Ausgangsdaten durch Netz iterieren
"""
for i in range(5):

    testout = cnn(test_input)
    #Netz trainieren und evaluieren
    #print('Shape des testout nach Durchlauf des Netzes:', testout.shape)
    #print('Shape des test_input:', test_input.shape)


    #Fehlerfunktion definieren und Fehler ausgeben
    criterion = nn.MSELoss()
    loss = criterion(testout, ziel_out)
    print('Loss:', loss)

    #Veränderungen der Gradienten zurücksetzen
    #cnn.zero_grad()
    #Loss durch Backpropagation laufen lassen
    #loss.backward()
    #Einstellung des Optimizers, Parameter und Lernrate
    # optimizer = optim.Adam(cnn.parameters(), lr=0.10)
    # optimizer.step()

#Netz speichern (besser mit geringstem Loss)
torch.save(cnn, 'cnn.pt')"""
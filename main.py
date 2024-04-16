# Importy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/diplomovka")
writer.close()

# Nacitavanie random input datasetov
def load_random_datasets(random_data_indexes, reduced=False ,priecinok_cesta=""):
    dataset = np.genfromtxt(f"FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0) if not reduced else np.genfromtxt(f"FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0)[:, ::2]
    data_len = dataset[0, :].__len__()
    data_num = random_data_indexes.shape[1]
    datasets = np.zeros([data_num, 1, data_len*2])
    nacitavanie_string = "Nacitavanie inputov"
    # print(nacitavanie_string, end="")
    for index, random_data_index in enumerate(random_data_indexes[0, :]):
        if index % 10 == 0:
            if nacitavanie_string == "Nacitavanie inputov...":
                nacitavanie_string = "Nacitavanie inputov"
            else:
                nacitavanie_string += "."
        print(f"\r{index+1}/{data_num} {nacitavanie_string}", end="")
        dataset = np.genfromtxt(f"FRF_databaza/FRF_{random_data_index}.csv", delimiter=',', skip_header=0) if not reduced else np.genfromtxt(f"FRF_databaza/FRF_{random_data_index}.csv", delimiter=',', skip_header=0)[:, ::2]
        datasets[index, 0, :data_len] = dataset[0, :]
        datasets[index, 0, data_len:] = np.log(dataset[1, :])  # Logaritmovanie y hodnoty
    print(f"\r{index+1}/{data_num} Inputy nacitane")
    return datasets


# Nacitanie random verification datasetov
def load_random_verification_dataset(random_data_indexes, priecinok_cesta = ""):
    dataset = np.genfromtxt(f"VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{1}.csv", delimiter=',', skip_header=0)
    data_len = dataset[0, :].__len__()
    data_num = random_data_indexes.shape[1]
    datasets = np.zeros([data_num, 1, data_len*2])
    nacitavanie_string = "Nacitavanie outputov"
    print(nacitavanie_string, end="")
    for index, random_data_index in enumerate(random_data_indexes[0, :]):
        if index % 10 == 0:
          if nacitavanie_string == "Nacitavanie outputov...":
              nacitavanie_string = "Nacitavanie outputov"
          else:
              nacitavanie_string += "."
        print(f"\r{index+1}/{data_num} {nacitavanie_string}", end="")
        dataset = np.genfromtxt(f"VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{random_data_index}.csv", delimiter=',', skip_header=0)
        datasets[index, 0, :data_len] = dataset[0, :]
        datasets[index, 0, data_len:] = dataset[1, :]
    print(f"\r{index+1}/{data_num} Overovacie outputy nacitane")
    return datasets

# Nacitanie dat na verifikaciu
def load_verification_datasets(data_index, reduced=False ,priecinok_cesta = ""):
    input_dataset = np.genfromtxt(f"FRF_databaza/FRF_{data_index}.csv", delimiter=',', skip_header=0) if not reduced else np.genfromtxt(f"FRF_databaza/FRF_{data_index}.csv", delimiter=',', skip_header=0)[:, ::2]
    data_len = input_dataset[0, :].__len__()
    verification_input = np.zeros([1, 1, data_len*2])
    verification_input[0, 0, :data_len] = input_dataset[0, :]
    verification_input[0, 0, data_len:] = np.log(input_dataset[1, :])
    output_dataset = np.genfromtxt(f"VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{data_index}.csv", delimiter=',', skip_header=0)
    data_len = output_dataset[0, :].__len__()
    verification_output = np.zeros([1, 2, data_len*2])
    verification_output[0, 0, :data_len] = output_dataset[0, :]
    verification_output[0, 0, data_len:] = output_dataset[1, :]
    return verification_input, verification_output

# Vytvaranie nahodnych batch-ov datasetov
def create_random_choice(num_of_data_in_batch, num_of_datasets):
    mylist = np.linspace(1, num_of_datasets, num_of_datasets, dtype=int).tolist()
    i = 0
    num_of_batches = num_of_datasets//num_of_data_in_batch
    random_datasets = np.zeros([num_of_batches, 1, num_of_data_in_batch], dtype=int)
    dataset_index = 0
    while True:
        temp = random.choice(mylist)
        mylist.remove(temp)
        # print(f"{i} temp:", temp)
        random_datasets[dataset_index, 0, i] = temp
        i += 1
        if i >= num_of_data_in_batch:
            dataset_index += 1
            i = 0
        if dataset_index >= num_of_batches:
            break
    return random_datasets

# Fukcia na vypocet casu
def calc_time(time_left):
    hod = f"0{int(time_left // (60*60))}" if len(str(int(time_left // (60*60)))) < 2 else int(time_left // (60*60))
    min = f"0{int((time_left % (60*60)) // 60)}" if len(str(int(time_left % (60*60) // 60))) < 2 else int((time_left % (60*60)) // 60)
    sec = f"0{int(time_left % (60*60) % 60)}" if len(str(int(time_left % (60*60) % 60))) < 2 else int(time_left % (60*60) % 60)
    return hod, min, sec

# Definicia Modelu/Neuralnej Siete
class VUF_odhadovac(nn.Module):
    def __init__(self, n_input_layers, n_output_layers):
        super(VUF_odhadovac, self).__init__()
        # Zadefinovanie vrstiev
        self.sequential_layers = nn.Sequential(
            nn.Linear(n_input_layers, 156),
            nn.ReLU(),
            nn.Linear(156, 78),
            nn.ReLU(),
            nn.Linear(78, 39),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(39, n_output_layers) # Output vrstva

    def forward(self, x):
        # Definicia forward propagation
        x = self.sequential_layers(x)
        x = self.output_layer(x)  # Posledna vrstva ma linernu aktivacnu funkciu
        return x

# TESTOVANIE FUNKCII
# temp = load_random_datasets(np.array([[1]]), True)


# ----------------------------------------------------------------------------------------------------------------------
# Nacitanie dat --------------------------------------------------------------------------------------------------------
# start_data  -> cislo PRVYCH dat intervalu na ktorom sa bude trenovat
# end_data    -> cislo POSLEDNYCH dat intervalu na ktorom sa bude trenovat
# num_of_data_in_batch -> pocet nahodnych datasetov vchadzjucich do trenovacieho procesu
start_data, end_data = (1, 100)
num_of_data_in_batch = 100
# Pocet epoch
epochs = 350
learning_rate = 0.001
reduced = True
model_name = f"model_{epochs}e_{num_of_data_in_batch}b_{end_data}d.pth"
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

print(f"Nacitane - pocet trenovacich datasetov: {end_data}\n\
         - pocet datasetov v batch-i:   {num_of_data_in_batch}\n\
         - pocet epoch:                 {epochs}"\
      if (end_data % num_of_data_in_batch == 0) or end_data < num_of_data_in_batch else\
      "Zadane hodnoty nie su spravne")

# Vyber poctu neuronov input vrstvy
example_input = load_random_datasets(np.array([[1]]), reduced)
n_input_layers = example_input.shape[2]
n_output_layers = 4

# Inicializacia Modelu
model = VUF_odhadovac(n_input_layers, n_output_layers)

# Loss & Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
torch.manual_seed(111)  # Zapnut ak chcem reprodukovatelne data
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Zobrazenie modelu v tensorboarde
writer.add_graph(model, torch.from_numpy(example_input).type(torch.float32))
writer.close()
# sys.exit()

# INFO
print(f"Pocet neuronov vo vstupnej vrstve:  {n_input_layers}")
print(f"Pocet neuronov vo vystupnej vrstve: {n_output_layers}")
print("\nInicializacia modelu prebehla uspesne")

# Trenovacia funkcia s random datami
total_iterations = epochs*end_data
iterations_left = epochs * end_data
times = list()
for epoch in range(epochs):
    # Nahodne rozhadzanie dat do batch-ov
    batches = create_random_choice(num_of_data_in_batch, end_data)
    if epoch >= 1:
        print("+"*50)
        print(f"Verifikacia modelu na datach c.{batches[0, 0 ,0]}")
        print("+"*50)
        verification_input, verification_output = load_verification_datasets(batches[0, 0, 0], reduced)
        print(f"Odhadnute hodnoty:  {model(torch.from_numpy(verification_input).type(torch.float32)).detach()[0, 0, :].tolist()}")
        print(f"Simulovane hodnoty: {verification_output[0, 0, :].tolist()}\n")
    # Nacitavanie jednotlivych batch-ov
    for batch_index, batch in enumerate(batches):
        print("-"*50)
        print(f"Epocha: {epoch+1}/{epochs}, Batch: {batch_index+1}/{batches.shape[0]}")
        print("-"*50)
        timer_load = time.time()
        input_datasets = load_random_datasets(batch, reduced)
        verification_datasets = load_random_verification_dataset(batch)
        timer_load = time.time() - timer_load
        # Trenovanie nacitaneho batchu
        print("Trenovanie:")
        timer_batch_train = time.time()
        for index, dataset in enumerate(input_datasets):
            timer_start = time.time()

            input_data = torch.from_numpy(dataset[0, :]).type(torch.float32)
            verification_data = torch.from_numpy(verification_datasets[index, 0, :]).type(torch.float32)
            outputs = model(input_data)
            loss = criterion(outputs, verification_data)

            optimizer.zero_grad() # Vynulovanie gradientu
            loss.backward()
            optimizer.step()
            # Zapisanie do tensorboardu
            iterations_left -= 1
            writer.add_scalar("loss", loss, total_iterations - iterations_left)
            if times.__len__() > 100:
                times.pop(0)
            times.append(time.time() - timer_start)
            if epoch+1 == epochs:
                time_left = round((epochs*end_data - (epoch*end_data + batch_index*num_of_data_in_batch + index)) * np.mean(times))
            else:
                time_left = round(((epochs*end_data - (epoch*end_data + batch_index*num_of_data_in_batch + index)) * np.mean(times)) + timer_load*(epochs-(epoch+1)))
            hod, min, sec = calc_time(time_left)
            print(f"\rOdhadovany cas vypoctu: {hod}h {min}m {sec}s", end="")
        print("\n")
    # writer.add_scalar('loss-epoch',)
    print("+"*50)
    print(f"Verifikacia modelu na datach c.{batches[0, 0 ,0]}")
    print("+"*50)
    verification_input, verification_output = load_verification_datasets(batches[0, 0, 0], reduced)
    print(f"Odhadnute hodnoty:  {model(torch.from_numpy(verification_input).type(torch.float32)).detach()[0, 0, :].tolist()}")
    print(f"Simulovane hodnoty: {verification_output[0, 0, :].tolist()}\n")
    print("Trenovanie dokoncene!")
    writer.close()

torch.save(model, model_name)

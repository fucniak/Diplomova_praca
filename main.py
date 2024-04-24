# Importy
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, os
# import sys
import random
# from torch.optim.lr_scheduler import ReduceLROnPlateau  # Dynamicky learning rate - step scheduler

# Prepnutie na GPU alebo CPU vypocet
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Pouzitie GPU
    print("Pouzitie GPU.")
else:
    device = torch.device("cpu")     # Pouzitie CPU ak CUDA nie je dostupna
    print("Pouzitie CPU.")


def data_preload(start_data, end_data, reduced, priecinok_cesta=f"{os.getcwd()}\\data_more"):
    """
    Nacitavanie vsetkych dat naraz do jedneho dictionary; pod jednym indexom je FRF, VUF, tlmenie a index dat
    :param start_data:
    :param end_data:
    :param reduced: Redukovanie FRF dat na polovicu (if True)
    :param priecinok_cesta:
    :return: dictionary so vsetkymi nacitanymi datami
    """
    datasets = {}
    # Allokuje priestor pre FRF
    frf_data_len = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0)[0, :].__len__() if not reduced else np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{1}.csv", delimiter=',', skip_header=0)[0, ::2].__len__()

    # Vytvori dictionary so vsetkymi datami s priradenym indexovanim
    for i, data_index in enumerate(range(start_data, end_data + 1)):
        print(f"\rNacitavanie dat: {i}/{end_data-start_data+1}", end="")
        datasets[f"{i}"] = {}
        datasets[f"{i}"]["data_index"] = data_index
        # Dictionary FRF
        frf_dataset = np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{data_index}.csv", delimiter=',',skip_header=0) if not reduced else np.genfromtxt(f"{priecinok_cesta}/FRF_databaza/FRF_{data_index}.csv", delimiter=',', skip_header=0)[:, ::2]
        # frf_data[0, :frf_data_len] = frf_dataset[0, :]
        # frf_data[0, frf_data_len:] = np.log(frf_dataset[1, :])  # Logaritmovanie y hodnoty
        datasets[f"{i}"]["frf"] = np.concatenate((frf_dataset[0, :], np.log(frf_dataset[1, :]))).reshape(1, -1)

        # Dictionary VUF
        datasets[f"{i}"]["vuf"] = np.array([np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{data_index}.csv", delimiter=',', skip_header=0)[0, :]])

        # Dictionary Tlmenie
        datasets[f"{i}"]["tlmenie"] = np.array([np.genfromtxt(f"{priecinok_cesta}/VUT_&_pomerne_tlmenie/vut_pomerne_tlmenie_{data_index}.csv", delimiter=',', skip_header=0)[1, :]])

    print("\rData nacitane!")
    return datasets


def accuracy_test(output, verification, by_one=False):
    output = output.cpu().detach().numpy()
    verification = verification.cpu().detach().numpy()
    out_1_acuraccy = (1 - abs(output[0, 0] - verification[0, 0]) / verification[0, 0]) * 100
    out_2_acuraccy = (1 - abs(output[0, 1] - verification[0, 1]) / verification[0, 1]) * 100
    # print("\nout_1", output[0,0], "out_2", output[0,1])
    # print("ver_1", verification[0,0], "ver_2", verification[0, 1])
    # print("1_accuracy", out_1_acuraccy)
    # print("2_accuracy", out_2_acuraccy)
    return (out_1_acuraccy+out_2_acuraccy)/2 if not by_one else round(out_1_acuraccy, 2), round(out_2_acuraccy, 2)


# Vytvaranie nahodnych batch-ov datasetov
# def create_random_choice(num_of_data_in_batch, num_of_datasets):
#     mylist = np.linspace(1, num_of_datasets, num_of_datasets, dtype=int).tolist()
#     i = 0
#     num_of_batches = num_of_datasets//num_of_data_in_batch
#     random_datasets = np.zeros([num_of_batches, 1, num_of_data_in_batch], dtype=int)
#     dataset_index = 0
#     while True:
#         temp = random.choice(mylist)
#         mylist.remove(temp)
#         # print(f"{i} temp:", temp)
#         random_datasets[dataset_index, 0, i] = temp
#         i += 1
#         if i >= num_of_data_in_batch:
#             dataset_index += 1
#             i = 0
#         if dataset_index >= num_of_batches:
#             break
#     return random_datasets


def calc_time(time_left):
    """
    Premeni dany pocet sekund na tvar HH:MM:SS
    :param time_left: Cas v sekundach
    :return: hodiny, minuty, secundy
    """
    hod = f"0{int(time_left // (60*60))}" if len(str(int(time_left // (60*60)))) < 2 else int(time_left // (60*60))
    min = f"0{int((time_left % (60*60)) // 60)}" if len(str(int(time_left % (60*60) // 60))) < 2 else int((time_left % (60*60)) // 60)
    sec = f"0{int(time_left % (60*60) % 60)}" if len(str(int(time_left % (60*60) % 60))) < 2 else int(time_left % (60*60) % 60)
    return hod, min, sec
class VUF_odhadovac(nn.Module):
    def __init__(self, n_input_layers, n_output_layers, h1, h2, h3, h4):
        super(VUF_odhadovac, self).__init__()
        # Zadefinovanie vrstiev
        self.sequential_layers = nn.Sequential(
            nn.Linear(n_input_layers, h1))

        self.output_layer = nn.Linear(h1, n_output_layers) # Output vrstva

    def forward(self, x):
        # Definicia forward propagation
        x = self.sequential_layers(x)
        x = self.output_layer(x)  # Posledna vrstva ma linernu aktivacnu funkciu
        return x

# Definicia Modelu/Neuralnej Siete
# class VUF_odhadovac(nn.Module):
#     def __init__(self, n_input_layers, n_output_layers, h1, h2, h3, h4):
#         super(VUF_odhadovac, self).__init__()
#         # Zadefinovanie vrstiev
#         self.sequential_layers = nn.Sequential(
#             nn.Linear(n_input_layers, h1),
#             nn.ReLU(),
#             nn.Linear(h1, h2),
#             nn.ReLU(),
#             nn.Linear(h2, h3),
#             nn.ReLU(),
#             nn.Linear(h3, h4),
#             nn.ReLU())
#         self.output_layer = nn.Linear(h4, n_output_layers) # Output vrstva
#
#     def forward(self, x):
#         # Definicia forward propagation
#         x = self.sequential_layers(x)
#         x = self.output_layer(x)  # Posledna vrstva ma linernu aktivacnu funkciu
#         return x

if __name__ == "__main__":
    priecinok_cesta = f"{os.getcwd()}\\data"
    # ----------------------------------------------------------------------------------------------------------------------
    # Nacitanie dat --------------------------------------------------------------------------------------------------------
    # start_data  -> cislo PRVYCH dat intervalu na ktorom sa bude trenovat
    # end_data    -> cislo POSLEDNYCH dat intervalu na ktorom sa bude trenovat
    # num_of_data_in_batch -> pocet nahodnych datasetov vchadzjucich do trenovacieho procesu
    start_data, end_data = (1, 30000)
    ver_start_data, ver_end_data = (30001, 31000)
    epochs = 500                        # Pocet epoch
    reduced = True                      # Zredukuje vstupne data na polovicu a tym sa zmensi aj pocet neur. v input vrstve
    output_option = "vuf"   # "tlmenie"
    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    print(f"Nacitane - pocet trenovacich datasetov: {end_data}\n\
             - Trenovanie:                  {output_option}\n\
             - pocet epoch:                 {epochs}\n\
             - Redukcia dat:                {'Ano' if reduced else 'Nie'}\n")

    # Vyber poctu neuronov input vrstvy
    example_input = data_preload(1,1,reduced)
    n_input_layers = example_input["0"]["frf"].shape[1]
    n_output_layers = 2

    # ----------------------------------------------------------------------------------------------------------------------
    # Inicializacia Modelu -------------------------------------------------------------------------------------------------
    model_type = "VUF"                  # Trenovanie bude na VUF alebo pomerne tlmenie
    learning_rate = 0.001              # Learning rate
    h1, h2, h3, h4 = 32, 312, 156, 78          # Velkost jednotlivych vrstiev v modeli NN
    layers_num = 3                      # Celkovy pocet vrstiev
    # Nastavenie ukladania
    model_hyperparameters = f"model_{str(learning_rate).replace('.', 'dot')}lr_{epochs}e_{end_data}d"
    model_parameters = f"model_{model_type}_{layers_num}l_{n_input_layers}il_{h1}_{n_output_layers}ol"
    model_name = "model.pth"
    model = VUF_odhadovac(n_input_layers, n_output_layers, h1, h2, h3, h4).to(device) # Inicializacia modelu
    # Loss & Optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    torch.manual_seed(111)  # Zapnut ak chcem reprodukovatelne data
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    optimizer.zero_grad()
    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # INFO
    print(f"Pocet neuronov vo vstupnej vrstve:  {n_input_layers}")
    print(f"Pocet neuronov vo vystupnej vrstve: {n_output_layers}")
    print("\nInicializacia modelu prebehla uspesne")

    # Trenovacia funkcia s random datami
    total_iterations = epochs * end_data
    iterations_left = epochs * end_data
    hod, min, sec = "nan", "nan", "nan"

    timer_epoch = list()

    # Definicia premnnych pre vypocet straty v ramci epoch na trenovacich datach
    epoch_loss_train_list = list()
    epoch_loss_train = 0.0

    # Definicia premnnych pre vypocet presnosti v ramci epoch na trenovacich datach
    epoch_accuracy_train_list = list()
    epoch_accuracy_train = 0.0

    # Definicia premnnych pre vypocet straty v ramci epoch na verifikacnych datach
    epoch_loss_var_list = list()

    # Definicia premnnych pre vypocet presnosti v ramci epoch na verifikacnych datach
    epoch_accuracy_var_list = list()


    preload_data = data_preload(start_data, end_data, True)
    preload_verification_data = data_preload(ver_start_data, ver_end_data, reduced)
    print('\nTrenovanie:')
    shuffle_indexes = list(range(end_data-start_data))

    for epoch in range(epochs):
        # Prepnutie modelu do stavu trenovania
        model.train()

        # Zapnutie casovaca jednej epochy
        timer_start = time.time()

        # Vytvorenie rozhadzania indexov dat
        random.shuffle(shuffle_indexes)

        # Trenovaci cyklus na rozhadzanych datach
        for i, data_index in enumerate(shuffle_indexes):
            optimizer.zero_grad() # Vynulovanie gradientu
            # Nacitanie inputov a porovnavacich dat
            input_data = torch.from_numpy(preload_data[f"{data_index}"]["frf"]).type(torch.float32).to(device)
            sim_output = torch.from_numpy(preload_data[f"{data_index}"][output_option]).type(torch.float32).to(device)
            outputs = model(input_data)

            # Vypocet starty (loss) a backpropagation
            loss = criterion(outputs, sim_output)
            loss.backward()
            optimizer.step()

            # Vypocet zostavajucich iteracii
            iterations_left -= 1

            # Pripocitanie straty jednej epochy na trenovacich datach
            epoch_loss_train += loss.item()

            # Pripocitanie accuracy jednej epochy na trenovacich datach
            epoch_accuracy_train += accuracy_test(outputs, sim_output)

            # Vypisanie epochy a zostavajuceho casu vypoctu
            print(f"\rEpocha: {epoch+1}/{epochs}, Zostava: {iterations_left} iter, Odhadovany cas vypoctu: {hod}h {min}m {sec}s\t", end="")

        # Vypocet priemernej straty jednej epochy train data
        epoch_loss_train_list.append(epoch_loss_train / end_data)
        epoch_loss_train = 0.0

        # Vypocet priemernej presnosti jednej epochy train data
        epoch_accuracy_train_list.append(epoch_accuracy_train / end_data)
        epoch_accuracy_train = 0.0

        # Vypocet straty na verifikacnych datach
        ver_data_num = ver_end_data - ver_start_data
        epoch_loss_ver = 0.0
        epoch_accuracy_var = 0.0
        model.eval()    # prepnutie modelu do modu intereferencie
        with torch.no_grad():
            for i in range(ver_data_num):
                ver_input = torch.from_numpy(preload_verification_data[f"{i}"]["frf"]).type(torch.float32).to(device)
                ver_sim_output = torch.from_numpy(preload_verification_data[f"{i}"][output_option]).type(
                    torch.float32).to(device)
                ver_output = model(ver_input)
                epoch_loss_ver += criterion(ver_output, ver_sim_output).item()
                epoch_accuracy_var += accuracy_test(ver_output, ver_sim_output)
            epoch_loss_var_list.append(epoch_loss_ver / ver_data_num)
            epoch_accuracy_var_list.append(epoch_accuracy_var / ver_data_num)

        # Posun learning rate podla straty na verifikacnych datach,
        # zapisanie accuracy na verifikacnych datach jednej epochy
        # scheduler.step(epoch_loss_ver / ver_data_num)

        # Vypocet casu jendej epochy
        timer_epoch.append(time.time() - timer_start)
        hod, min, sec = calc_time((epochs-(epoch+1))*np.mean(timer_epoch))

    print("\n\nTrenovanie dokoncene!\n\n")

    # Vypocet accuracy na datach na ktorych nebol model trenovany
    print("Vypocet accuracy na verifikacnych datach:")
    ver_data_num = ver_end_data - ver_start_data
    ver_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(ver_data_num):
            ver_input = torch.from_numpy(preload_verification_data[f"{i}"]["frf"]).type(torch.float32).to(device)
            ver_sim_output = torch.from_numpy(preload_verification_data[f"{i}"][output_option]).type(torch.float32).to(device)
            ver_output = model(ver_input)
            ver_accuracy += accuracy_test(ver_output, ver_sim_output)

    print(f"\nPresnost modelu na verifikacnych datach: {round(ver_accuracy / ver_data_num, 2)}%")
    print('Oslava??')


    # Zobrazenie priemernej straty na epochach trenovacie data
    fig, ax = plt.subplots()
    ax.set_title(f"Strata - Trénovacie dáta")
    ax.plot(epoch_loss_train_list, label='Strata', color="red")
    ax.set_xlabel(f"Epochy")
    ax.grid()
    ax.legend()

    # Vypocet priemernej presnosti jednej epochy train data
    fig1, ax1 = plt.subplots()
    ax1.plot(epoch_accuracy_train_list, label='Presnosť', color="green")
    ax1.set_title(f"Priemerná presnosť - trénovacie dáta")
    ax1.set_xlabel(f"Epochy")
    ax1.grid()
    ax1.legend()

    # Zobrazenie priemernej straty na epochach trenovacie data
    fig2, ax2 = plt.subplots()
    ax2.set_title(f"Strata - Verifikačné dáta")
    ax2.plot(epoch_loss_var_list, label='Strata', color="red")
    ax2.set_xlabel(f"Epochy")
    ax2.grid()
    ax2.legend()

    # Zobrazenie priemernej straty na epochach trenovacie data
    fig3, ax3 = plt.subplots()
    ax3.set_title(f"Presnosť - Verifikačné dáta")
    ax3.plot(epoch_accuracy_var_list, label='Presnosť', color="green")
    ax3.set_xlabel(f"Epochy")
    ax3.grid()
    ax3.legend()


    # Ukladanie modelu a potrebnych informacii -----------------------------------------------------------------------------
    if not os.path.exists(f"models/{model_parameters}"):
        os.makedirs(f"models/{model_parameters}")
    if not os.path.exists(f"models/{model_parameters}/{model_hyperparameters}"):
        os.makedirs(f"models/{model_parameters}/{model_hyperparameters}")
    torch.save(model, f"models/{model_parameters}/{model_hyperparameters}/{model_name}")
    # torch.save(model, f"Layers_training/one_layer_32hn_500e_30000d")

    # # Ulozenie priebehu loss
    # with open(f"models/{model_parameters}/{model_hyperparameters}/loss.txt", 'w') as file:
    #     # Write each item on a new line
    #     for item in ver_accuracy_list_epoch:
    #         file.write(str(item) + '\n')

    fig.savefig(f"models/{model_parameters}/{model_hyperparameters}/loss_train.png", format='png', dpi=300, bbox_inches='tight')
    fig1.savefig(f"models/{model_parameters}/{model_hyperparameters}/accuracy_train.png", format='png', dpi=300, bbox_inches='tight')
    fig2.savefig(f"models/{model_parameters}/{model_hyperparameters}/loss_ver.png", format='png', dpi=300, bbox_inches='tight')
    fig3.savefig(f"models/{model_parameters}/{model_hyperparameters}/accuracy_ver.png", format='png', dpi=300, bbox_inches='tight')
    # ----------------------------------------------------------------------------------------------------------------------
    plt.show()

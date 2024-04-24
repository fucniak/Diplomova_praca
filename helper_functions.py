import numpy as np
import os, json
from Dynamic_parameter_identifier import *

def data_preload(start_data, end_data, reduced, priecinok_cesta=f"{os.getcwd()}\\data"):
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
    return (out_1_acuraccy+out_2_acuraccy)/2 if not by_one else (round(out_1_acuraccy, 2), round(out_2_acuraccy, 2))


def root_mean_squered_error(output, verification, by_one=False):
    output = output.cpu().detach().numpy()
    verification = verification.cpu().detach().numpy()
    out_1_rmse = np.sqrt((output[0, 0] - verification[0, 0])**2)
    out_2_rmse = np.sqrt((output[0, 1] - verification[0, 1])**2)
    return round((out_1_rmse + out_2_rmse) / 2,  2) if not by_one else (round(out_1_rmse, 2), round(out_2_rmse, 2))


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


def load_model(model_path):
    # Nacitanie parametrov modelu
    with open(f"{model_path}/model_information.json", "r") as file:
        model_information = json.load(file)

    # Pocet neuronov na vstupe
    input_neurons = model_information["model_parameters"]["layers"]["input_layer"]["neurons"]

    # Pocet skrytych vrstiev a pocet ich neuronov
    num_of_hidden_layers = model_information["model_parameters"]["layers"]["hidden_layers"]["number_of_layers"]
    hidden_layers_neurons = [model_information["model_parameters"]["layers"]["hidden_layers"][f"{i}"]["neurons"] for i in range(num_of_hidden_layers)]

    # Pocet neuronov na vystupe
    output_neurons = model_information["model_parameters"]["layers"]["output_layer"]["neurons"]

    # Inicializacia modelu
    model = Dynamic_parameter_identifier(input_neurons, hidden_layers_neurons, output_neurons)

    # Load the state dictionary
    model.load_state_dict(torch.load(f"{model_path}/model_weights.pth"))

    # Mod interferencie
    model.eval()

    return model, model_information



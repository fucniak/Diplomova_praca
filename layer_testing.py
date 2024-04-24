from helper_functions import *
import time
import random
import matplotlib.pyplot as plt

# Prepnutie na GPU alebo CPU vypocet
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Pouzitie GPU
    print("Pouzitie GPU.")
else:
    device = torch.device("cpu")     # Pouzitie CPU ak CUDA nie je dostupna
    print("Pouzitie CPU.")

# ----------------------------------------------------------------------------------------------------------------------
color_list = ["red", "green", "blue", "black", "purple", "orange", "brown", "grey", "pink", "yellow"]
model_name = "model_testing_1"
save_path = f"models/layers_training/{model_name}"
save_flag = True
# ----------------------------------------------------------------------------------------------------------------------
# Nacitanie dat --------------------------------------------------------------------------------------------------------
start_data, end_data = (1, 50)
ver_start_data, ver_end_data = (51, 60)

# Pocet modelov na vytvorenie
num_of_models = 3

# Pocet epoch
epochs = 100

# Zredukuje vstupne data na polovicu a tym sa zmensi aj pocet neur. v input vrstve
reduced = True

# Typ vystupu
output_option = "vuf"   # "tlmenie"

# Aktivacna funkcia
activation_function = "linear"

# Learning rate
learning_rate = 0.001

# Velkost jednotlivych vrstiev v modeli NN
hidden_layers = [5, 4, 2]

# Celkovy pocet vrstiev
layers_num = hidden_layers.__len__()

# Vyber poctu neuronov input vrstvy
example_input = data_preload(1, 1, reduced)
n_input_layers = example_input["0"]["frf"].shape[1]
n_output_layers = 2

# Prednacitanie trenovacich a verifikacnych dat
preload_data = data_preload(start_data, end_data, True)
preload_verification_data = data_preload(ver_start_data, ver_end_data, reduced)
shuffle_indexes = list(range(end_data-start_data))
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Proces tvorby novych modelov zo zadanych vstupnych parametrov
for model_index in range(num_of_models):
    # Vyber farby
    color = color_list[model_index]

    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print(f"Nacitane - pocet trenovacich datasetov:  {end_data}\n\
         - Trenovanie:                   {output_option}\n\
         - pocet epoch:                  {epochs}\n\
         - pocet neuronov v s. vrstvach: {hidden_layers}\n\
         - Redukcia dat:                 {'Ano' if reduced else 'Nie'}\n")

    # INICIALIZACIA MODELU
    model = Dynamic_parameter_identifier(n_input_layers, hidden_layers, n_output_layers).to(device) # Inicializacia modelu

    # Mean Squared Error Loss
    criterion = nn.MSELoss()

    # Zapnut ak chcem reprodukovatelne data
    seed = 111
    torch.manual_seed(seed)

    # Vyber optimizera
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    print("Trenovanie:")
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

        # Vypocet casu jendej epochy
        timer_epoch.append(time.time() - timer_start)
        hod, min, sec = calc_time((epochs-(epoch+1))*np.mean(timer_epoch))

    print("\n\nTrenovanie dokoncene!\n")

    # Vypocet accuracy na datach na ktorych nebol model trenovany
    print("Vypocet starty a presnosti:")
    ver_data_num = ver_end_data - ver_start_data + 1
    ver_accuracy = 0.0
    ver_loss = 0.0
    train_data_num = end_data - start_data + 1
    train_accuracy = 0.0
    train_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(ver_data_num):
            ver_input = torch.from_numpy(preload_verification_data[f"{i}"]["frf"]).type(torch.float32).to(device)
            ver_sim_output = torch.from_numpy(preload_verification_data[f"{i}"][output_option]).type(torch.float32).to(device)
            ver_output = model(ver_input)
            ver_loss_item = criterion(ver_output, ver_sim_output)
            ver_accuracy += accuracy_test(ver_output, ver_sim_output)
            ver_loss += ver_loss_item.item()
        for i in range(train_data_num):
            train_input = torch.from_numpy(preload_data[f"{i}"]["frf"]).type(torch.float32).to(device)
            train_sim_output = torch.from_numpy(preload_data[f"{i}"][output_option]).type(torch.float32).to(device)
            train_output = model(train_input)
            train_loss_item = criterion(train_output, train_sim_output)
            train_accuracy += accuracy_test(train_output, train_sim_output)
            train_loss += train_loss_item.item()

    ver_accuracy_mean = round(ver_accuracy / ver_data_num, 2)
    ver_loss_mean = round(ver_loss / ver_data_num, 2)
    train_accuracy_mean = round(train_accuracy / train_data_num, 2)
    train_loss_mean = round(train_loss / train_data_num, 2)
    print(f"Presnost modelu na verifikacnych datach: {ver_accuracy_mean}%")
    print(f"Strata modelu na verifikacnych datach: {ver_loss_mean}")
    print(f"Presnost modelu na trenovacich datach: {train_accuracy_mean}%")
    print(f"Strata modelu na trenovacich datach: {train_loss_mean}\n")

    # Zobrazenie priemernej straty na epochach trenovacie data
    fig, ax = plt.subplots()
    ax.set_title(f"Strata - Trénovacie dáta; {hidden_layers}")
    ax.plot(epoch_loss_train_list, label='Strata', color="red")
    ax.set_xlabel(f"Epochy")
    ax.grid()
    ax.legend()

    # Vypocet priemernej presnosti jednej epochy train data
    fig1, ax1 = plt.subplots()
    ax1.set_title(f"Priemerná presnosť - trénovacie dáta; {hidden_layers}")
    ax1.plot(epoch_accuracy_train_list, label='Presnosť', color="green")
    ax1.set_xlabel(f"Epochy")
    ax1.grid()
    ax1.legend()

    # Zobrazenie priemernej straty na epochach trenovacie data
    fig2, ax2 = plt.subplots()
    ax2.set_title(f"Strata - Verifikačné dáta; {hidden_layers}")
    ax2.plot(epoch_loss_var_list, label='Strata', color="red")
    ax2.set_xlabel(f"Epochy")
    ax2.grid()
    ax2.legend()

    # Zobrazenie priemernej straty na epochach trenovacie data
    fig3, ax3 = plt.subplots()
    ax3.set_title(f"Presnosť - Verifikačné dáta; {hidden_layers}")
    ax3.plot(epoch_accuracy_var_list, label='Presnosť', color="green")
    ax3.set_xlabel(f"Epochy")
    ax3.grid()
    ax3.legend()

    # ------------------------------------------------------------------------------------------------------------------
    # Ukladanie --------------------------------------------------------------------------------------------------------
    hidden_layers_dict = {"number_of_layers": hidden_layers.__len__()}
    for i, hidden_layer in enumerate(hidden_layers):
        hidden_layers_dict[f"{i}"] = {}
        hidden_layers_dict[f"{i}"]["neurons"] = hidden_layer
        hidden_layers_dict[f"{i}"]["activation_function"] = activation_function

    model_parameters = {
        "layers": {
            "input_layer": {
                "neurons": n_input_layers,
                "activation_function": activation_function},
            "hidden_layers": hidden_layers_dict,
            "output_layer": {
                "neurons": n_output_layers,
                "activation_function": activation_function}},
        "learning_rate": learning_rate,
        "optimizer": "ADAM",
        "loss": "MSE",
        "seed": seed,
        "reduced": reduced}

    model_informations = {
        "model_parameters": model_parameters,
        "training_process": {
            "output_option": output_option,
            "pretrained": None,
            "epochs": epochs,
            "train_data": {
                "start_data": start_data,
                "end_data": end_data,
                "loss": train_loss_mean,
                "accuracy": train_accuracy_mean
            },
            "verification_data": {
                "start_data": ver_start_data,
                "end_data": ver_end_data,
                "loss": ver_loss_mean,
                "accuracy": ver_accuracy_mean
            }
        }
    }

    if save_flag:
        save_path_full = f"{save_path}/model_{model_index}"
        print(f"Ukladanie do: {save_path_full}\n\n")
        # Vytvorenie priecinku
        if not os.path.exists(f"{save_path_full}"):
            os.makedirs(f"{save_path_full}")

        # Ulozenie vah modelu
        torch.save(model.state_dict(), f"{save_path_full}/model_weights.pth")

        # Ulozenie informacii o modele ako json
        with open(f"{save_path_full}/model_information.json", "w") as file:
            json.dump(model_informations, file, indent=4)

    # Ukladanie grafov
    fig.savefig(f"{save_path_full}/loss_train.png", format='png', dpi=300, bbox_inches='tight')
    fig1.savefig(f"{save_path_full}/accuracy_train.png", format='png', dpi=300, bbox_inches='tight')
    fig2.savefig(f"{save_path_full}/loss_ver.png", format='png', dpi=300, bbox_inches='tight')
    fig3.savefig(f"{save_path_full}/accuracy_ver.png", format='png', dpi=300, bbox_inches='tight')
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    hidden_layers = [i*2 for i in hidden_layers]

plt.show()

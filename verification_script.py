from helper_functions import *

# Cesta k priecinku kde je ulozeny model
model_path = "models/layers_training/model_testing_1/model_2"

# Nacitanie modelu
model, model_information = load_model(model_path)

# ----------------------------------------------------------------------------------------------------------------------
# Nacitanie dat --------------------------------------------------------------------------------------------------------
start_data, end_data = (1, 10)
ver_start_data, ver_end_data = (9501, 10000)
epochs = 5                        # Pocet epoch
reduced = model_information["model_parameters"]["reduced"]                   # Zredukuje vstupne data na polovicu a tym sa zmensi aj pocet neur. v input vrstve["
output_option = model_information["training_process"]["output_option"]
hidden_layers = model_information["model_parameters"]["layers"]["hidden_layers"]["number_of_layers"]
criterion = nn.MSELoss()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------")
print(f"Nacitany model")
print(f"Vstupne neurony: {model_information['model_parameters']['layers']['input_layer']['neurons']}")
print(f'Skryte neurony: {[model_information["model_parameters"]["layers"]["hidden_layers"][f"{i}"]["neurons"] for i in range(hidden_layers)]}')
print(f"Vystupne neurony: {model_information['model_parameters']['layers']['output_layer']['neurons']}")
print(f"Optimizer: {model_information['model_parameters']['optimizer']}")
print(f"Strata: {model_information['model_parameters']['loss']}")
print(f"Output moznost: {model_information['training_process']['output_option']}")
print(f"Trenovanie na datach: {model_information['training_process']['train_data']['start_data']} - {model_information['training_process']['train_data']['end_data']}")
print("--------------------------------------------------------------------")


# Verifikacia modelu
ver_data_num = ver_end_data - ver_start_data + 1
input_data = data_preload(ver_start_data, ver_end_data, reduced)
rmse = 0.0
model.eval()
with torch.no_grad():
    for i in range(ver_data_num):
        ver_input = torch.from_numpy(input_data[f"{i}"]["frf"]).type(torch.float32)
        ver_sim_output = torch.from_numpy(input_data[f"{i}"][output_option]).type(torch.float32)
        ver_output = model(ver_input)
        print(f"Data Index: FRF_{input_data[f'{i}']['data_index']}")
        print("Odhadovane hodnoty:", ver_output.detach().numpy())
        print("Simulovane hodnoty:", ver_sim_output.detach().numpy())
        print(f"Root Mean Squered Error: {root_mean_squered_error(ver_output, ver_sim_output)}")
        print(f"Presnost: {accuracy_test(ver_output, ver_sim_output, by_one=True)}%")
        print('\n')



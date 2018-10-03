import json
from utils.gann2 import Gann2
from utils.caseman import Caseman
from utils import tflowtools as tft
from possible_config import loss_functions, case_sets, hidden_activation_functions, output_activation_functions,optimizers


keys = list(case_sets.keys())
for (i, key) in enumerate(keys):
    print(i, ". " + key)

index = input("Hvilket datasett vil du teste? ")
file_name = keys[int(index)]

print(file_name)

with open("../json_files/" + file_name + ".json") as file:
    data = json.load(file)

case_name = data["case_name"]

data_set = case_sets[case_name](**data["case_config"])

cman = Caseman(lambda: data_set,
               vfrac=data["validation_fraction"],
               tfrac=data["test_fraction"],
               cfrac=data["case_fraction"])

gann = Gann2(caseman=cman,
             layer_sizes=data["layer_sizes"],
             top_k=data["top_k"],
             learning_rate=data["learning_rate"],
             init_weight_range=data["init_weight_range"],
             hidden_activation_function=hidden_activation_functions[data["hidden_activation_function"]],
             output_activation_function=output_activation_functions[data["output_activation_function"]],
             loss_function=loss_functions[data["loss_function"]],
             optimizer=optimizers[data["optimizer"]],
             minibatch_size=data["minibatch_size"])

sess = tft.gen_initialized_session()

gann.add_summary(0, "weights", ["avg"])
gann.add_summary(1, "output", ["avg"])

gann.run(sess,
         epochs=data["epochs"],
         validation_interval=data["validation_interval"],
         show_interval=data["show_interval"])
tft.close_session(sess)


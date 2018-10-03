import json
from utils.gann2 import Gann2
from utils.caseman import Caseman
from utils import tflowtools as tft
from possible_config import loss_functions, case_sets, hidden_activation_functions, output_activation_functions, optimizers



def main():
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

    gann.add_summary(gann.error, only_validation=False)
    gann.add_summary(gann.accuracy, only_validation=False)
    gann.add_dendrogram(0)
    gann.add_dendrogram(1)
    gann.add_dendrogram(2)
    gann.add_summary(0, "weights", "hist")
    gann.add_summary(1, "weights", ["avg", "max", "min"])
    gann.add_summary(2, "output", ["avg", "max", "min"])

    gann.add_fetched_var(0, "weights")
    gann.add_hinton(0, ["input", "weights", "output"])
    gann.add_hinton(["input", 2, "output", "target"], ["", "weights", "", ""])

    gann.run(sess,
             epochs=data["epochs"],
             validation_interval=data["validation_interval"],
             show_interval=data["show_interval"])
    tft.close_session(sess, True)
    print("Hei")


def visualization():
    print("0. Hinton-plot \n1. Dendogram \n2. Weights and biases \n3. Alle grafene")
    comand = input("Hva ønsker du å printe? ")
    with open("../json_files/" + comand + ".json") as file:
        data = json.load(file)

    if comand == "0":
        gann.add_hinton()

def fire_up():
    dir = "summary"
    cmd = "tensorboard --logdir=%s" % dir
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)


def stop(pipe):
    pipe.kill()
    os.system('rm -r ' + "summary")
    #os.killpg(os.getpgid(pipe.pid), signal.SIGTERM)


main()
time.sleep(1)
pipe = fire_up()

cont = 1
while cont:
    next = input("Neste?")
    stop(pipe)
    if next == "y":
        main()
        pipe = fire_up()
        time.sleep(1)
    else:
        cont = 0



cont = 1
while cont:
    main()
    new_round = input("Ønsker du å teste et annet datasett (y/n)? ")
    if new_round == "1":
        cont = 0

    vizualisation = input("Hvilke visualisering ønsker du å vise? ")
    if vizualisation == "0":
        pass
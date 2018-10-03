import subprocess
import time
import os
import json
from utils.gann2 import Gann2
from utils.caseman import Caseman
from utils import tflowtools as tft
from possible_config import loss_functions, case_sets, hidden_activation_functions, output_activation_functions, \
    optimizers


def main(data, visuals):
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
    if visuals:
        for key, args in visuals:
            if key == "hinton":
                for arg in args:
                    gann.add_hinton(*arg)
            elif key == "dendrogram":
                for arg in args:
                    gann.add_dendrogram(*arg)
            elif key == "summary":
                for arg in args:
                    gann.add_summary(*arg)

    gann.add_summary(gann.error, only_validation=False)
    gann.add_summary(gann.accuracy, only_validation=False)
    gann.run(sess,
             epochs=data["epochs"],
             validation_interval=data["validation_interval"],
             show_interval=data["show_interval"])
    tft.close_session(sess, False)


def fire_up():
    dir = "summary"
    cmd = "tensorboard --logdir=%s" % dir
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)


def stop(pipe):
    pipe.kill()
    os.system('rm -r ' + "summary")
    # os.killpg(os.getpgid(pipe.pid), signal.SIGTERM)


def ask_for_visualization():
    tb = input("Vise i tensorboard (y/n)?")
    if tb == "y":
        with open("../json_files/visualization.json") as file:
            data = json.load(file)
        _visuals = []
        print("Hvilke visualiseringer vil du ha?")
        choose_all = input("Vise alle (y/n)?")
        if choose_all == "y":
            return list(data.items())
        for key, value in data.items():
            choose = input("%s?" % key)
            if choose == "y":
                _visuals.append((key, value))
        return _visuals
    else:
        return False


def ask_for_case_set():
    keys = list(case_sets.keys())
    for (i, key) in enumerate(keys):
        print("%d. %s" % (i, key))

    index = input("Hvilket datasett vil du teste? ")
    file_name = keys[int(index)]

    print(file_name)

    with open("../json_files/" + file_name + ".json") as file:
        return json.load(file)


cont = 1
next = "y"
pipe = None
while next == "y":
    data = ask_for_case_set()
    visuals = ask_for_visualization()
    main(data, visuals)
    if visuals:
        pipe = fire_up()
        time.sleep(3)
    next = input("Neste (y/n)?")
    if pipe is not None:
        stop(pipe)

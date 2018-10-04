import tensorflow as tf
from utils import tflowtools as tft
from utils.caseman import Caseman
from utils.gann import Gann
import json
from utils.image_plotter import draw_dendrogram

from possible_config import loss_functions, optimizers, hidden_activation_functions, output_activation_functions

data_set = tft.gen_all_parity_cases(10)

cman = Caseman(lambda: data_set, 0.1, 0.1)

dims = [10, 16, 6, 2]

"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""

"""
"""
prefered_accuracy = 0.95
print("Acuuracy should be: ", prefered_accuracy)

"""
features =[[-0.05113626240760806, -0.08198335676823273, -0.09726780245682415, 0.5176314124050275, -0.11989226213198079, -0.1335414344107227, 0.5278523020662838, -0.11836879720730308, -0.11879371025854416, 0.38786227091395636, 0.7662959751933802, 0.5890134650523908, 0.0994803096003741, -0.12075002770900319, 0.5690670307198606, 0.5583303958252508], [-0.08130556019349688, -0.06764883948941679, -0.09536359862615307, 0.286751362029541, -0.08409017791189498, -0.08872195122739537, 0.4202334919195086, -0.08579079763039661, -0.08863274861164468, 0.14426120915123997, 0.4370042778806927, 0.3036757225904111, 0.17359484954291338, -0.07234892332610246, 0.27912204137905783, 0.3713378854973664], [-0.11110979417630396, -0.07348958533985621, -0.09836027693833414, 0.3695638847661805, -0.1229015650126492, -0.11284838841813204, 0.6017595633170925, -0.09016313159334387, -0.10367305540874892, 0.2369582307251119, 0.5443466704182531, 0.33206984719832244, 0.15656971150073662, -0.10313927879991075, 0.3972471251010178, 0.4864845940406859], [-0.15090889718344666, -0.1553794547771018, -0.13991179065464407, 0.5747676407830835, -0.18914707765849856, -0.18801045079283324, 0.9073881597832634, -0.1829617511320931, -0.17517771317336142, 0.437954400482883, 1.004419716260659, 0.8278391446082807, 0.25625278615623565, -0.16024875509305048, 0.7834665759894193, 0.7899621814651966], [-0.142689987144119, -0.1498436693611981, -0.17083632594146492, 0.8501444316534051, -0.17413112335641642, -0.23249910871175739, 0.9543380618491266, -0.19377026269338873, -0.23166096079911927, 0.5581521141897203, 1.2334979222636326, 0.8155970749636018, 0.09204861202532504, -0.18094915408849824, 0.8915708510106473, 0.9582750935534742], [-0.15409078316851113, -0.1490973681026336, -0.17196202047716624, 0.6499863321802486, -0.19858782024034444, -0.22141402348888023, 1.0426516109575055, -0.19500072578812389, -0.21551337990892538, 0.4715954649717853, 1.312396856222576, 0.7702286865457539, 0.031795864823949485, -0.16235482602372653, 0.9122210960832018, 1.0528320809928298], [-0.1005962410569993, -0.11394892368003201, -0.12694497375665753, 0.5800488145764223, -0.12865083347678966, -0.14147746060479435, 0.518968302274752, -0.11502675298286867, -0.14153693730876152, 0.42214792597881956, 0.7138365978389638, 0.5141689896371546, 0.04335721856169491, -0.1307096309429854, 0.5014582812743538, 0.555648563664737], [-0.1383228591354982, -0.1235314677279502, -0.15370037886662963, 0.5259083857895197, -0.16134139100673722, -0.1565687525907204, 0.705431129023455, -0.11539846153646452, -0.148853489956994, 0.3459839924106155, 0.83481868978252, 0.3894766101448114, 0.0630036798389135, -0.09080176478912212, 0.5634461685141596, 0.7319036849662902], [-0.07079200707419224, -0.10810817782959257, -0.12394829544447646, 0.4972362918397828, -0.08983944637603541, -0.11735102341405768, 0.33744223087716796, -0.11065441901992142, -0.1264966305116573, 0.3294509044049476, 0.6064942053014033, 0.4857748650292433, 0.06038235660387166, -0.0999192754691771, 0.3833331975523938, 0.4405018551214176]]
labels = ['0.01.00.00.01.01.00.01.00.00.0', '0.00.00.01.00.00.01.00.00.01.0', '0.00.01.00.00.00.01.01.00.00.0', '0.01.00.01.01.01.01.01.01.00.0', '1.01.01.00.01.01.01.01.00.01.0', '0.01.01.00.01.01.01.01.01.01.0', '1.00.01.00.01.00.00.01.00.00.0', '1.00.01.00.00.00.00.01.01.01.0', '1.00.00.01.01.00.00.00.00.01.0']


image = draw_dendrogram(features, labels)
image2 = draw_dendrogram(features, labels)
image3 = draw_dendrogram(features, labels)
images = [image, image2, image3]

dendrogram_placeholder = tf.placeholder(tf.string, None, "dendrogram_placeholder")

decoded = tf.image.decode_png(dendrogram_placeholder, channels=3)
decoded = tf.expand_dims(decoded, 0)
dendrogram_summary = tf.summary.image("dendro_image", decoded)

label_placeholder = tf.placeholder(tf.string, None, "label_placeholder")

sess = tft.gen_initialized_session("test_summary")
fetched_vars = {
    "halla": [],
    "hei": [],
    "sums": tf.summary.merge_all()
}
for i in range(3):
    vals = sess.run(fetched_vars, feed_dict={dendrogram_placeholder: images[i]})
    print(vals)
tft.close_session(sess)

"""

gann = Gann(dims, cman, top_k=1,
            loss_function=lambda labels, predictions: tf.losses.mean_squared_error(labels=labels,
                                                                                    predictions=predictions),
            output_activation_function=tf.nn.softmax,
            hidden_activation_function=tf.nn.leaky_relu,
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0035,
            minibatch_size=10
            )

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

gann.run(sess, 250, validation_interval=10, show_interval=40)
tft.close_session(sess)

# show in hinton:
#    input, output i hvert layer, og output

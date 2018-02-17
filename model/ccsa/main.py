import model.ccsa.Initialization as Initialization
import logging
from keras.layers import Activation, Dropout, Dense
from keras.layers import Input, Lambda
from keras.models import Model

from data_helper.DataBuilderPan import DataBuilderPan

logging.basicConfig(level=logging.INFO)
# Creating embedding function. This corresponds to the function g in the paper.
# You may need to change the network parameters.
model_g = Initialization.dg_cnn()

input_shape = (1, 100, 1)
input_ak = Input(shape=input_shape)
input_au = Input(shape=input_shape)
input_bk = Input(shape=input_shape)
input_bu = Input(shape=input_shape)

# number of classes for digits classification
nb_classes = 2

# Loss = (1-alpha)Classification_Loss + (alpha)CSA
alpha = .25

# Having two streams. One for source and one for target.
processed_a = model_g(input_ak, input_au)
processed_b = model_g(input_bk, input_bu)

# Creating the prediction function. This corresponds to h in the paper.
out1 = Dropout(0.5)(processed_a)
out1 = Dense(nb_classes)(out1)
out1 = Activation('softmax', name='classification')(out1)

distance = Lambda(Initialization.euclidean_distance, output_shape=Initialization.eucl_dist_output_shape, name='CSA')(
    [processed_a, processed_b])
model = Model(inputs=[[input_ak, input_au], [input_bk, input_bu]], outputs=[out1, distance])
model.compile(loss={'classification': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
              optimizer='adadelta',
              loss_weights={'classification': 1 - alpha, 'CSA': alpha})

tr_pairs = Initialization.load_data()
Acc = Initialization.training_the_model(model, tr_pairs, epochs=5, batch_size=2)
print(('Best accuracy is {}.'.format(Acc)))

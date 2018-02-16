import Initialization
import logging
from keras.layers import Activation, Dropout, Dense
from keras.layers import Input, Lambda
from keras.models import Model

logging.basicConfig(level=logging.INFO)
# Creating embedding function. This corresponds to the function g in the paper.
# You may need to change the network parameters.
model_g=Initialization.dg_cnn()

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
model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': Initialization.contrastive_loss},
              optimizer='adadelta',
              loss_weights={'classification': 1 - alpha, 'CSA': alpha})

print('Domain Adaptation Task: ' + domain_adaptation_task)
# let's create the positive and negative pairs using row data.
# pairs will be saved in ./pairs directory

Initialization.Create_Pairs(domain_adaptation_task,repetition,sample_per_class)
Acc=Initialization.training_the_model(model,domain_adaptation_task,repetition,sample_per_class)
print(('Best accuracy for {} target sample per class and repetition {} is {}.'.format(sample_per_class,repetition,Acc )))










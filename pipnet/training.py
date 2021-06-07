import tensorflow as tf
import numpy as np
import time
import os
from pipnet.pip import PIP
from datetime import datetime
from pipnet.utils import pairwise_dist
from pipnet.utils import generate_and_save_images
from pipnet.utils import print_classification_report
from pipnet.utils import print_prediction_report
from pipnet.utils import plot_loss


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['AUTOGRAPH_VERBOSITY'] = '10'
tf.autograph.set_verbosity(0)
print("TensorFlow version: {}".format(tf.__version__))
print('Tenserflow CUDA is available: {}'.format(tf.test.is_built_with_cuda()))
print("Eager execution: {}".format(tf.executing_eagerly()))

MSE = tf.keras.losses.MeanSquaredError()
CCE = tf.keras.losses.CategoricalCrossentropy()
weights = tf.Variable(np.array([1.0, 1.0, 1.0, 1.0]),dtype=tf.float32)
train_losses = []
test_losses = []
loss = tf.keras.metrics.Mean()

now = datetime.now().time()

@tf.function
def draw(x):
    """
    Draws a random element proportional to their values.
    
    # Arguments:
        x: list of floats, weight ratios \lambda
    # Returen:
        index of selected element
    """
    p = x / tf.math.reduce_sum(x)
    idx = tf.random.categorical(tf.math.log([p]), 1)[0][0]
    return idx

@tf.function
def update_weights(losses, tau, epsilon):
    """
    If the selected loss is less than the threshold (tau), 
    reward the term by decreasing its weight; otherwise, increase it.
    
    # Arguments:
        losses: list of floats, (E, M, R1, R2)
        tau: threshold
        epsilon: weight learning rate
    # Returen:
        update the selected weight
    """
    idx = draw(weights)
    l = losses[idx]
    if l < tau:
        weights[idx].assign(weights[idx] * tf.math.pow(1.+epsilon, -1.0*l))
    else:
        weights[idx].assign(weights[idx] * tf.math.pow(1.-epsilon, -1.0*l))


@tf.function
def compute_loss(model, x, y, img):
    """
    Computes the PIP loss function as described in the paper
    
    # Arguments:
        model: training model
        x: input data
        y: labels
        img: dedicated picture for each label
    # Returen:
        PIP loss, generator loss, classifier loss, R1, R2
    """
    z = model.encode(x)
    x_logit = model.generate(z)
    y_logits, prototype_distances, feature_vector_distances, prototype, a = model.classification(z)
    l_ae = MSE(img, x_logit)
    l_c = CCE(y, y_logits)
    r_e = tf.reduce_mean(tf.reduce_min(feature_vector_distances, axis = 1), name='error_1')
    r_c = tf.reduce_mean(tf.reduce_min(prototype_distances, axis = 1), name='error_2')
    total_error = (weights[0]*l_ae) + (weights[1]*l_c) + (weights[2]*r_e) + (weights[3]*r_c)
    return total_error, l_ae, l_c, r_e, r_c

@tf.function
def compute_apply_gradients(model, x, y, img, optimizer):
    """
    Apply the gradiant
    """
    with tf.GradientTape() as tape:
        loss, _, _, _, _ = compute_loss(model, x, y, img)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def fit(model, inputs, targets, target_pic, val_inputs, val_targets, val_target_pic, 
        optimization_lr, weight_lr, treshold, weightUpdate_wait, early_stopping, batch_size, num_epochs,file_name):
    """
    PIP Training
    """
    TRAIN_BUF = 2*inputs.shape[0]
    TEST_BUF = inputs.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets, target_pic)).shuffle(TRAIN_BUF).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets, val_target_pic)).shuffle(TEST_BUF).batch(batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimization_lr, amsgrad=True)

    prev_loss = 0.0
    weight_update = True
    best_loss = tf.constant(10000.0)
    stopping_step = 0
    p = 0
    should_stop = False

    history_path = './training_history/'+file_name+'/'+str(now)
    if not os.path.exists('./training_history'):
        os.mkdir('./training_history') 
    if not os.path.exists('./training_history/'+file_name):
        os.mkdir('./training_history/'+file_name)
    if not os.path.exists(history_path):
        os.mkdir(history_path)
    checkpoint_directory = history_path+"/training_checkpoint" 
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    for epoch in range(1, num_epochs):
        start_time = time.time()
        # compute loss and apply gradient
        for train_x, train_y, train_img in train_dataset:
            compute_apply_gradients(model, train_x, train_y, train_img, optimizer)
        end_time = time.time()
        losstotal, l_ae, l_c, r_e, r_c = compute_loss(model, train_x, train_y, train_img)
        loss(losstotal)
        train_error = loss.result()
        train_losses.append(train_error)
        # updating weight
        if train_error < prev_loss:
            p = p + 1
            if p == weightUpdate_wait:
                weight_update = False
                print('> Updating weight terminated.')
        else: 
            p = 0
            prev_loss = train_error
        if weight_update:
            losses = tf.stack([l_ae, l_c, r_e, r_c])
            update_weights(losses, treshold, weight_lr)
        # compute validation error
        for test_x, test_y, test_img in test_dataset:
            h, _, _, _, _ = compute_loss(model, test_x, test_y, test_img)
            loss(h)
        test_error = loss.result()
        test_losses.append(test_error)
        # print the model performance
        print('Epoch: {}, Train set error: {}, Test set error: {}, '
        'time elapse for current epoch {}'.format(epoch,
                                                    train_error,
                                                    test_error,
                                                    end_time - start_time))
        # save the  image of prototypes at each epoch
        generate_and_save_images(model, epoch, history_path)
        # early stopping
        if epoch >= early_stopping:
            if test_error < best_loss:
                stopping_step = 0
                best_loss = test_error
            else:
                stopping_step += 1
            if stopping_step == early_stopping:
                should_stop = True
                checkpoint.save(file_prefix=checkpoint_prefix)
                break
    plot_loss(train_losses, test_losses, history_path)

def classification_report(model, inputs, target, class_names, file_name):
    """
    Save PIP result
    """
    history_path = './training_history/'+file_name+'/'+str(now)
    encoding = model.encode(inputs)
    y_pred, _, _, _, a = model.classification(encoding)
    print_classification_report(y_pred, target, class_names, history_path)
    print_prediction_report(tf.argmax(y_pred,axis=1), tf.argmax(target,axis=1), a, history_path)
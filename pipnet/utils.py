import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
import pandas as pd


def generate_and_save_images(model, epoch, file_path):
    """
    Transform the prototype to picture via PIP's generator
    """
    file_path = file_path+'/pictorial_prototype/'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    var = [v for v in model.trainable_variables if v.name == "prototype_feature_vectors:0"][0]
    predictions = model.generate(var).numpy()
    fig = plt.figure(figsize=(12,6))
    for i in range(predictions.shape[0]):
        plt.subplot(6, 3, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig(file_path+'/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight')
    plt.close()

def report_to_df(report):
    """
    Make the classification report ready to be saved
    """
    report = [x.split(' ') for x in report.split('\n')]
    header = [' ']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:]:
        row = [value for value in row if value!='']
        if row!=[]:
            if row[0] == 'accuracy':
                values.append(['accuracy', '', '', row[1], row[2]])
            elif row[0] == 'macro':
                values.append(['macro avg', row[2], row[3], row[4], row[5]])
            elif row[0] == 'weighted':
                values.append(['weighted avg', row[2], row[3], row[4], row[5]])
            else:
                values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df

def print_classification_report(y_pred, target, class_names, file_path):
    """
    Save the classification error
    """
    report = classification_report(tf.argmax(target,1), tf.argmax(y_pred,1), target_names=class_names)
    print(report)
    df = report_to_df(report)
    df.to_csv(file_path+'/classification_report.csv', sep=',',index=False)

def print_prediction_report(y_pred, target, a, file_path):
    """
    Save the model prediction and 'a' layer result
    """
    n_prototypes = a.shape[1]
    columns_names = ['target', 'y_pred']
    prototypes_name = ['a'+str(i) for i in range(n_prototypes)]
    columns_names = columns_names + prototypes_name
    d = tf.concat([tf.dtypes.cast(tf.stack([target, y_pred],axis=1),tf.float32), a], axis=1).numpy()
    result = pd.DataFrame(d, columns=columns_names)
    result.to_csv(file_path+'/prediction_report.csv', sep=',',index=False)

def plot_loss(train_losses, test_losses, file_path):
    """
    Plot the training and testing loss
    """
    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='testing loss')
    plt.legend()
    plt.ylabel("Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.savefig(file_path+'/train_test_loss_results')


@tf.function
def pairwise_dist(X, Y):
    '''
    Pairwise distance between two vectors
    # Argument:
                X: ndtensor
                Y: mdtensor
    # Returnes:
                D: (m x n) matrix
    '''
    # squared norm of each vector
    XX = tf.reduce_sum(tf.square(X), 1)
    YY = tf.reduce_sum(tf.square(Y), 1)
    # XX is a row vector and YY is a column vector
    XX = tf.reshape(XX, [-1, 1])
    YY = tf.reshape(YY, [1, -1])
    return XX + YY - 2 * tf.matmul(X, tf.transpose(Y))


@tf.function
def normalize(x_i):
    """
    Normalize the vector
    """
    min_i = tf.math.reduce_min(x_i)
    max_i = tf.math.reduce_max(x_i)
    x_i_normalized = (x_i - min_i) / (max_i - min_i)
    return x_i_normalized
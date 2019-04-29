import gzip
import pickle
import os
import multiprocessing as mp
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection

import src.config as config


class DataEnv(object):
    def __init__(
            self, training_data_file_path, test_data_file_path, parameter_data_path, _final_class_num,
            _batch_size, _num_epochs):
        raw_training_data = pd.read_pickle(training_data_file_path, compression="gzip")
        raw_test_data = pd.read_pickle(test_data_file_path, compression="gzip")
        with gzip.open(parameter_data_path) as f_in:
            parameter_data_dict = pickle.load(f_in)
        self.pca = parameter_data_dict['pca']
        self.feature_num = parameter_data_dict['select_var_num']
        self.label_class_num = _final_class_num

        self._train_data, self._test_data = sklearn.model_selection.train_test_split(
            raw_training_data, test_size=0.1)
        self._final_test_data = raw_test_data
        self.label_single_col = config.label_col
        # self.label_cols = pd.Index(["{}_{}".format(config.label_col, i + 1) for i in range(_final_class_num)])
        self.feature_cols = raw_training_data.columns.difference([self.label_single_col])

        self.tf_feature_columns = [tf.feature_column.numeric_column(key=key) for key in self.feature_cols]
        # self.tf_label_columns = [tf.feature_column.numeric_column(key=key) for key in self.label_cols]

        self.train_size = len(self._train_data)
        self.test_size = len(self._test_data)
        self.batch_size = _batch_size
        self.num_epochs = _num_epochs

        # self.predicting_input_data = (raw_predicting_input_data - self.input_feature_mean) / self.input_feature_std
        # self.predicting_output_file_path = predicting_output_data_file_path

    def train_input_fn(self):
        raw_train_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(self._train_data[self.feature_cols]), self._train_data[self.label_single_col]))
        train_dataset = raw_train_dataset.shuffle(self.train_size).repeat(self.num_epochs).batch(self.batch_size)
        return train_dataset

    def test_input_fn(self):
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(self._test_data[self.feature_cols]), self._test_data[self.label_single_col]))
        test_dataset = raw_test_dataset.batch(self.batch_size)
        return test_dataset

    def predict_test_fn(self):
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(
            dict(self._final_test_data[self.feature_cols]))
        test_feature_dataset = raw_test_dataset.batch(self.batch_size)
        return test_feature_dataset

    def loss_of_predict_test(self, prediction_iterator):
        label_data_array = np.array(self._final_test_data[self.label_single_col])
        prediction_array = list(prediction_iterator)
        correct_num = np.count_nonzero(label_data_array == prediction_array)
        # for prediction_value, label_data_value in zip(prediction_iterator, label_data_series):
        #
        return correct_num / len(label_data_array)


# Define the neural network
def neural_network(features, params):
    # TF Estimator input is a dict, in case of multiple inputs
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    if params['l2_strength'] > 1e-5:
        l2_regularizer = tf.contrib.layers.l2_regularizer(params['l2_strength'])
    else:
        l2_regularizer = None
    hidden_list = params['hidden_list']
    final_classes = params['result_dim']
    keep_rate = params['keep_rate']
    current_layer = input_layer
    for hidden_unit in hidden_list:
        current_layer = tf.layers.dense(
            current_layer, units=hidden_unit, activation=tf.nn.relu, kernel_regularizer=l2_regularizer)
        if keep_rate < 0.999:
            dropout_layer = tf.nn.dropout(current_layer, keep_prob=keep_rate)
            current_layer = dropout_layer
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(
        current_layer, final_classes, kernel_regularizer=l2_regularizer)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode, params):
    # Build the neural network
    final_logits = neural_network(features, params)

    # If prediction mode, early return
    # predicted_result = tf.nn.softmax(final_logits)
    predicted_result = tf.argmax(final_logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions=predicted_result)
            # mode, predictions=predicted_result[:, tf.newaxis])  # Reshape to [-1, 1] form

    # label_tensor = tf.feature_column.input_layer(labels, params['label_columns'])
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels, final_logits)

    # Define loss and optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        starter_learning_rate = params['learning_rate']
        decay_steps = params['decay_steps']
        decay_rate = params['decay_rate']
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=params['momentum'],
            use_nesterov=params['nesterov'])
        train_op = optimizer.minimize(loss_op, global_step=global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op)

    else:
        # Evaluate the accuracy of the model
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_result)
        tf.summary.scalar('accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, eval_metric_ops={'accuracy': accuracy})


def training_and_testing(argv):
    # test_running = config.test_running

    # Parameters
    num_steps = 40000
    batch_size = 64
    train_epochs = 1000

    # display_step = 1000
    save_checkpoints_steps = 2000
    keep_checkpoint_max = 3

    data_reader = DataEnv(
        config.reduced_train_data_frame_file, config.reduced_test_data_frame_file, config.output_parameter_file,
        config.activity_num, batch_size, train_epochs)

    # Network Parameters
    params = {
        'hidden_list': [32, 16],
        'input_dim': data_reader.feature_num,
        'result_dim': data_reader.label_class_num,
        'keep_rate': 1,
        'learning_rate': 5e-3,
        'decay_rate': 1 - 5e-5,
        'decay_steps': 1,
        'momentum': 0.8,
        'nesterov': True,
        'l1_strength': 0,
        'l2_strength': 0,
        'feature_columns': data_reader.tf_feature_columns,
        'label_column': data_reader.label_single_col
    }

    train_spec = tf.estimator.TrainSpec(
        input_fn=data_reader.train_input_fn, max_steps=num_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=data_reader.test_input_fn, throttle_secs=5)

    # Build the Estimator
    run_config = tf.estimator.RunConfig(
        keep_checkpoint_max=keep_checkpoint_max, save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=1000)
    model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=config.nn_model_dir, config=run_config, params=params)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    test_predictions = model.predict(input_fn=data_reader.predict_test_fn)
    print("Accuracy of test set: {:.3f}".format(data_reader.loss_of_predict_test(test_predictions)))
    # predictions = model.predict(input_fn=data_reader.predict_input_fn)
    # data_reader.output_predictions(predictions)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=training_and_testing, argv=None)
    # running_config = tf.ConfigProto(
    #     device_count={"CPU": 7},  # limit to num_cpu_core CPU usage
    #     inter_op_parallelism_threads=1,
    #     intra_op_parallelism_threads=4,
    #     log_device_placement=True)
    # with tf.Session(config=running_config) as sess:
    #      sess.run


def parallel_main():
    cluster = {"master": ["localhost:2222"],
               "worker": ["localhost:2223", "localhost:2224", "localhost:2225"],
               "ps": ["localhost:2221"]}
    p_list = []
    for job_name, task_list in cluster.items():
        for index, task in enumerate(task_list):
            current_task_dict = {'type': job_name, 'index': index}
            current_tf_config = {'cluster': cluster, 'task': current_task_dict}
            os.environ['TF_CONFIG'] = json.dumps(current_tf_config)
            p = mp.Process(target=main, args=())
            p_list.append(p)
            p.start()

    for p in p_list:
        p.join()


if __name__ == '__main__':
    # main()
    parallel_main()

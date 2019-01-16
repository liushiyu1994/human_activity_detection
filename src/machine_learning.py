import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection

import config as config


class DataEnv(object):
    def __init__(
            self, training_data_file_path, predicting_input_data_file_path,
            predicting_output_data_file_path, _batch_size, label_cols_list, _num_epochs):
        raw_training_data = pd.read_pickle(
            training_data_file_path, compression="gzip").astype(np.float32)
        raw_predicting_input_data = pd.read_pickle(
            predicting_input_data_file_path, compression="gzip").astype(np.float32)
        mean_value = raw_training_data.mean()
        std_value = raw_training_data.std()
        norm_training_data = (raw_training_data - mean_value) / std_value

        self._train_data, self._test_data = sklearn.model_selection.train_test_split(
            norm_training_data, test_size=0.1)
        self.label_cols = pd.Index(label_cols_list)
        self.feature_cols = raw_training_data.columns.difference(self.label_cols)
        self.input_feature_mean = mean_value[self.feature_cols]
        self.input_feature_std = std_value[self.feature_cols]
        self.output_label_mean = mean_value[self.label_cols]
        self.output_label_std = std_value[self.label_cols]
        self.tf_feature_columns = [tf.feature_column.numeric_column(key=key) for key in self.feature_cols]
        self.tf_label_columns = [tf.feature_column.numeric_column(key=key) for key in self.label_cols]

        self.train_size = len(self._train_data)
        self.test_size = len(self._test_data)
        self.batch_size = _batch_size
        self.num_epochs = _num_epochs

        self.predicting_input_data = (raw_predicting_input_data - self.input_feature_mean) / self.input_feature_std
        self.predicting_output_file_path = predicting_output_data_file_path

    def train_input_fn(self):
        raw_train_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(self._train_data[self.feature_cols]), self._train_data[self.label_cols]))
        train_dataset = raw_train_dataset.shuffle(self.train_size).repeat(self.num_epochs).batch(self.batch_size)
        return train_dataset

    def test_input_fn(self):
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(self._test_data[self.feature_cols]), self._test_data[self.label_cols]))
        test_dataset = raw_test_dataset.batch(self.batch_size)
        return test_dataset

    def predict_test_fn(self):
        raw_test_dataset = tf.data.Dataset.from_tensor_slices(
            dict(self._test_data[self.feature_cols]))
        test_feature_dataset = raw_test_dataset.batch(self.batch_size)
        return test_feature_dataset

    def loss_of_predict_test(self, prediction_iterator):
        loss_score = 0
        count = 0
        for prediction_array in prediction_iterator:
            # prediction_array = np.array([result_dict['predictions'][0]])
            # print(prediction_array[0])
            correct_array = np.array(self._test_data[self.label_cols])
            loss_score += np.sqrt(np.mean((prediction_array - correct_array)**2))
            count += 1
        return loss_score / count

    def predict_input_fn(self):
        raw_predict_dataset = tf.data.Dataset.from_tensor_slices(
            dict(self.predicting_input_data[self.feature_cols]))
        predict_dataset = raw_predict_dataset.batch(self.batch_size)
        return predict_dataset

    def output_predictions(self, prediction_iterator):
        raw_predicting_output = np.array(
            [result for result in prediction_iterator])
        output_data_frame = pd.DataFrame(
            raw_predicting_output, index=self.predicting_input_data.index, columns=self.label_cols)
        output_data_frame = output_data_frame * self.output_label_std + self.output_label_mean
        output_data_frame.to_pickle(self.predicting_output_file_path, compression="gzip")


# Define the neural network
def neural_net(features, params):
    # TF Estimator input is a dict, in case of multiple inputs
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    l2_regularizer = tf.contrib.layers.l2_regularizer(params['l2_strength'])
    layer_1 = tf.layers.dense(
        input_layer, params['hidden_1_dim'], activation=tf.nn.leaky_relu, kernel_regularizer=l2_regularizer)
    layer_2 = tf.layers.dense(
        layer_1, params['hidden_2_dim'], activation=tf.nn.leaky_relu, kernel_regularizer=l2_regularizer)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(
        layer_2, params['result_dim'], kernel_regularizer=l2_regularizer)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode, params):
    # Build the neural network
    final_regression = neural_net(features, params)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions=final_regression)

    # label_tensor = tf.feature_column.input_layer(labels, params['label_columns'])
    loss_op = tf.losses.mean_squared_error(labels, final_regression)

    # Define loss and optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op)

    else:
        # Evaluate the accuracy of the model
        square_error_op = tf.metrics.mean_squared_error(labels=labels, predictions=final_regression)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, eval_metric_ops={'mean_squared_error': square_error_op})


def training_and_testing(argv):
    # Parameters
    num_steps = 1000
    batch_size = 128
    train_epochs = 1e5

    display_step = 100
    save_step = 1000
    epochs_per_eval = 2

    data_reader = DataEnv(
        config.training_output_file, config.predicting_features_file,
        config.predicting_labels_file, batch_size, config.aa_feature_list, epochs_per_eval)

    # Network Parameters
    params = {
        'hidden_1_dim': 32,  # 1st layer number of neurons
        'hidden_2_dim': 32,  # 2nd layer number of neurons
        'input_dim': len(data_reader.feature_cols),  # Currently is 36
        'result_dim': len(data_reader.label_cols),  # Currently is 18
        'learning_rate': 0.1,
        'l1_strength': 0,
        'l2_strength': 0.001,
        'feature_columns': data_reader.tf_feature_columns,
        'label_columns': data_reader.tf_label_columns
    }

    train_spec = tf.estimator.TrainSpec(input_fn=data_reader.train_input_fn, max_steps=train_epochs // epochs_per_eval)
    eval_spec = tf.estimator.EvalSpec(input_fn=data_reader.test_input_fn, throttle_secs=600)

    # Build the Estimator
    run_config = tf.estimator.RunConfig(
        keep_checkpoint_max=3, save_checkpoints_steps=save_step)
    model = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=config.nn_model_dir, config=run_config, params=params)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    test_predictions = model.predict(input_fn=data_reader.predict_test_fn)
    print("MAE of test set: {:.3f}".format(data_reader.loss_of_predict_test(test_predictions)))
    predictions = model.predict(input_fn=data_reader.predict_input_fn)
    data_reader.output_predictions(predictions)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # args, unparsed = parser.parse_known_args()
    # tf.app.run(main=training_and_testing, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=training_and_testing, argv=None)


if __name__ == '__main__':
    main()

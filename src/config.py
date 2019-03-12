import platform

data_direct = "./data"
training_data_direct = "{}/train".format(data_direct)
test_data_direct = "{}/test".format(data_direct)
output_data_direct = "{}/output".format(data_direct)

feature_direct = "./UCI HAR Dataset"
feature_list_file = "{}/features.txt".format(feature_direct)

train_data_x_file = "{}/X_train.txt".format(training_data_direct)
train_data_y_file = "{}/y_train.txt".format(training_data_direct)
test_data_x_file = "{}/X_test.txt".format(test_data_direct)
test_data_y_file = "{}/y_test.txt".format(test_data_direct)

train_data_frame_file = "{}/train_output_pickle.gz".format(output_data_direct)
reduced_train_data_frame_file = "{}/reduced_train_output_pickle.gz".format(output_data_direct)
test_data_frame_file = "{}/test_output_pickle.gz".format(output_data_direct)
reduced_test_data_frame_file = "{}/reduced_test_output_pickle.gz".format(output_data_direct)
output_parameter_file = "{}/output_parameter_pickle.gz".format(output_data_direct)
complete_stat = "{}/complete_stat.xlsx".format(output_data_direct)

nn_model_dir = "{}/nn_model".format(data_direct)

label_col = 'label'
activity_num = 6


if platform.node() == 'BaranLiu-PC':
    test_running = True
else:
    test_running = False

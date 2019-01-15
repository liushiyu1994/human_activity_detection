data_direct = "./data"
nutrition_category_file = "{}/NUTR_DEF.txt".format(data_direct)
nutrition_data_file = "{}/NUT_DATA.txt".format(data_direct)
output_pickle_gz_file = "{}/output_pickle.gz".format(data_direct)
complete_stat = "{}/complete_stat.xlsx".format(data_direct)
filtered_output_pickle_gz_file = "{}/output_pickle_filtered.gz".format(data_direct)
filtered_stat = "{}/filtered_stat.xlsx".format(data_direct)
important_features_file = "{}/important_features.xlsx".format(data_direct)
training_output_file = "{}/output_pickle_selected.gz".format(data_direct)
predicting_features_file = "{}/predicting_features_pickle.gz".format(data_direct)
predicting_labels_file = "{}/predicting_labels_pickle.gz".format(data_direct)

aa_feature_list = [str(i) for i in range(501, 519)]
nn_model_dir = "{}/nn_model".format(data_direct)

import gzip
import pickle

import pandas as pd
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np

import src.config as config


def collect_feature_name(feature_file):
    unique_feature_list = []
    with open(feature_file) as f_in:
        for line in f_in:
            new_line = line.strip()
            index, feature_name = new_line.split()
            unique_feature_list.append("{}-{}".format(index, feature_name))
    return unique_feature_list


def read_data(df_column_name, feature_data_file, label_data_file):
    feature_df = pd.read_csv(feature_data_file, delim_whitespace=True, names=df_column_name)
    label_df = pd.read_csv(label_data_file, delim_whitespace=True, names=[config.label_col])
    label_df -= 1
    # if dummy:
    #     label_dummy_df = pd.get_dummies(label_df[config.label_col], prefix=config.label_col)
    #     result_df = pd.concat([feature_df, label_df, label_dummy_df], axis=1, sort=False)
    # else:
    #     result_df = pd.concat([feature_df, label_df], axis=1, sort=False)
    return feature_df, label_df


def pca_reduce(data_frame, pca, final_var_num):
    new_array = pca.transform(data_frame)
    return new_array[:, :final_var_num]


def statistics(data_frame, output_file):
    # PCA result
    # label_df = data_frame.loc[:, [config.label_col]]
    # feature_df = data_frame.drop(config.label_col, axis=1)
    pca = sklearn.decomposition.PCA()
    # new_feature_array = pca.fit_transform(feature_df, label_df)
    new_feature_array = pca.fit_transform(data_frame)
    # print(pca.explained_variance_ratio_[:10])
    select_var_num = np.count_nonzero(pca.explained_variance_ratio_ > 1e-3)
    print(select_var_num)
    print(np.sum(pca.explained_variance_ratio_[:select_var_num]))
    fig, ax = plt.subplots()
    ax.scatter(new_feature_array[:, 1], new_feature_array[:, 2])
    fig, ax = plt.subplots()
    bar_plot_variance_ratio = pca.explained_variance_ratio_[1:30]
    x_loc = np.arange(0, len(bar_plot_variance_ratio))
    ax.bar(x_loc, bar_plot_variance_ratio)
    output_dict = {'select_var_num': select_var_num, 'pca': pca}
    with gzip.open(output_file, 'w') as f_out:
        pickle.dump(output_dict, f_out)
    # plt.show()
    return pca, select_var_num


def process_data(train_feature_df, test_feature_df):
    pca, select_var_num = statistics(train_feature_df, config.output_parameter_file)
    train_array = pca_reduce(train_feature_df, pca, select_var_num)
    test_array = pca_reduce(test_feature_df, pca, select_var_num)
    column_list = ["PC_{}".format(i + 1) for i in range(train_array.shape[1])]
    reduced_train_feature_df = pd.DataFrame(train_array, columns=column_list)
    reduced_test_feature_df = pd.DataFrame(test_array, columns=column_list)
    return reduced_train_feature_df, reduced_test_feature_df


def read_and_formulate_data():
    unique_feature_name_list = collect_feature_name(config.feature_list_file)
    train_feature_df, train_label_df = read_data(
        unique_feature_name_list, config.train_data_x_file, config.train_data_y_file)
    test_feature_df, test_label_df = read_data(
        unique_feature_name_list, config.test_data_x_file, config.test_data_y_file)
    reduced_train_feature_df, reduced_test_feature_df = process_data(train_feature_df, test_feature_df)
    train_complete_df = pd.concat([train_feature_df, train_label_df], axis=1, sort=False)
    train_reduced_df = pd.concat([reduced_train_feature_df, train_label_df], axis=1, sort=False)
    test_complete_df = pd.concat([test_feature_df, test_label_df], axis=1, sort=False)
    test_reduced_df = pd.concat([reduced_test_feature_df, test_label_df], axis=1, sort=False)
    train_complete_df.to_pickle(config.train_data_frame_file, compression='gzip')
    train_reduced_df.to_pickle(config.reduced_train_data_frame_file, compression='gzip')
    test_complete_df.to_pickle(config.test_data_frame_file, compression='gzip')
    test_reduced_df.to_pickle(config.reduced_test_data_frame_file, compression='gzip')


def check_repeat(repeat_col_dict, data_x_file):
    line_count = 100
    first_name = list(repeat_col_dict.keys())[0]
    with open(data_x_file) as f_in:
        for line in f_in:
            if line_count < 0:
                break
            number_list = line.split()
            repeat_col_list = repeat_col_dict[first_name]
            print([number_list[repeat_col] for repeat_col in repeat_col_list])
            line_count -= 1


def main():
    read_and_formulate_data()
    # process_data()
    plt.show()


if __name__ == '__main__':
    main()

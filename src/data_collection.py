import pandas as pd

import src.config as config


def construct_nutrition_id_dict(category_file):
    nutrition_pair_list = []
    with open(category_file) as f_in:
        for line in f_in:
            raw_list = line.split('^')
            nutrition_id = raw_list[0].strip('~')
            nutrition_name = raw_list[3].strip('~')
            nutrition_pair_list.append((nutrition_id, nutrition_name))
    return nutrition_pair_list


def read_data(df_columns, data_file):
    def add_more_nutrition(_current_food_series, _nutrition_id, _mean_value, _source_count):
        source_filter = 0
        if _source_count >= source_filter:
            _current_food_series[_nutrition_id] = _mean_value

    series_list = []
    current_food_series = pd.Series(name="00000")
    with open(data_file) as f_in:
        for line in f_in:
            raw_list = line.split('^')
            food_id = raw_list[0].strip('~')
            nutrition_id = raw_list[1].strip('~')
            mean_value = float(raw_list[2])
            source_count = int(raw_list[3])
            if food_id != current_food_series.name:
                series_list.append(current_food_series)
                current_food_series = pd.Series(name=food_id)
            add_more_nutrition(current_food_series, nutrition_id, mean_value, source_count)
    result_df = pd.DataFrame(series_list, columns=df_columns)
    return result_df


def statistics(data_frame, nutrition_pair_list, output_file):
    raw_list = list(zip(*nutrition_pair_list))
    nutrition_series = pd.Series(raw_list[1], index=raw_list[0])
    count_series = data_frame.count()
    total_food_num = len(data_frame)
    ratio_series = count_series / total_food_num

    final_df = pd.DataFrame({'Num': count_series, 'Ratio': ratio_series, 'Name': nutrition_series})
    final_df.to_excel(output_file)


def select_features(
        data_frame, output_training_file, output_predicting_features_file,
        output_predicting_labels_file):
    threshold_ratio = 0.55
    count_series = data_frame.count()
    total_food_num = len(data_frame)
    above_threshold_feature_index = count_series[count_series > total_food_num * threshold_ratio].index
    aa_feature_columns = pd.Index(config.aa_feature_list)
    important_features_df = pd.read_excel(config.important_features_file)
    important_features_columns = pd.Index(
        [str(i) for i in important_features_df[important_features_df['Chosen'] > 0].index])
    input_feature_columns = important_features_columns.difference(aa_feature_columns)

    training_data_frame = data_frame[important_features_columns].dropna()
    print(training_data_frame)
    training_data_frame.to_pickle(output_training_file, compression='gzip')

    data_frame_with_all_features = data_frame[input_feature_columns].dropna()
    training_data_index = training_data_frame.index
    predicting_data_index = data_frame_with_all_features.index.difference(training_data_index)
    predicting_input_data_frame = data_frame.loc[predicting_data_index][input_feature_columns]
    predicting_output_data_frame = data_frame.loc[predicting_data_index][aa_feature_columns]
    print(predicting_input_data_frame)
    print(predicting_output_data_frame)
    predicting_input_data_frame.to_pickle(output_predicting_features_file, compression='gzip')
    predicting_output_data_frame.to_pickle(output_predicting_labels_file, compression='gzip')


def process_data():
    # nutrition_pair_list = construct_nutrition_id_dict(config.nutrition_category_file)
    complete_df = pd.read_pickle(config.output_pickle_gz_file)
    # statistics(complete_df, nutrition_pair_list, config.complete_stat)

    # filtered_df = pd.read_pickle(config.filtered_output_pickle_gz_file)
    # statistics(filtered_df, nutrition_pair_list, config.filtered_stat)
    select_features(
        complete_df, config.training_output_file, config.predicting_features_file,
        config.predicting_labels_file)


def read_and_formulate_data():
    nutrition_pair_list = construct_nutrition_id_dict(config.nutrition_category_file)
    nutrition_id_list = list(zip(*nutrition_pair_list))[0]
    result_df = read_data(nutrition_id_list, config.nutrition_data_file)
    result_df.to_pickle(config.output_pickle_gz_file, compression='gzip')


def main():
    # read_and_formulate_data()
    process_data()


if __name__ == '__main__':
    main()

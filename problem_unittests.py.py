
# coding: utf-8

# In[ ]:


import re
import pandas as pd
import operator 

# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto 
# the dataframe - uses stratified random sampling (with seed) to sample by task & plagiarism amount 

# Use function to label datatype for training 1 or test 2 
def create_datatype(df, train_value, test_value, datatype_var, compare_dfcolumn, operator_of_compare, value_of_compare,
                    sampling_number, sampling_seed):
    # Subsets dataframe by condition relating to statement built from:
    # 'compare_dfcolumn' 'operator_of_compare' 'value_of_compare'
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns = [datatype_var])

    # Prints counts by task and compare_dfcolumn for subset df
    #print("\nCounts by Task & " + compare_dfcolumn + ":\n", df_subset.groupby(['Task', compare_dfcolumn]).size().reset_index(name="Counts") )

    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value

    # Performs stratified random sample of subset dataframe to create new df with subset values 
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(lambda x: x.sample(min(len(x), sampling_number), random_state = sampling_seed))
    df_sampled = df_sampled.drop(columns = [datatype_var])
    # Sets all datatype to value for test_value for df_sampled
    df_sampled.loc[:, datatype_var] = test_value

    # Prints counts by compare_dfcolumn for selected sample
    #print("\nCounts by "+ compare_dfcolumn + ":\n", df_sampled.groupby([compare_dfcolumn]).size().reset_index(name="Counts") )
    #print("\nSampled DF:\n",df_sampled)

    # Labels all datatype_var column as train_value which will be overwritten to 
    # test_value in next for loop for all test cases chosen with stratified sample
    for index in df_sampled.index: 
        # Labels all datatype_var columns with test_value for straified test sample
        df_subset.loc[index, datatype_var] = test_value

    #print("\nSubset DF:\n",df_subset)
    # Adds test_value and train_value for all relevant data in main dataframe
    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value based upon 
        # stratified test sample and subset of df
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]

    # returns nothing because dataframe df already altered 

def train_test_dataframe(clean_df, random_seed=100):

    new_df = clean_df.copy()

    # Initialize datatype as 0 initially for all records - after function 0 will remain only for original wiki answers
    new_df.loc[:,'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)

    # creating a dictionary of categorical:numerical mappings for plagiarsm categories
    mapping = {0:'orig', 1:'train', 2:'test'} 

    # traversing through dataframe and replacing categorical data
    new_df.Datatype = [mapping[item] for item in new_df.Datatype] 

    return new_df


# helper function for pre-processing text given a file
def process_file(file):
    # put text in all lower case letters 
    all_text = file.read().lower()

    # remove all non-alphanumeric chars
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)

    return all_text


def create_text_column(df, file_directory='data/'):
    '''Reads in the files, listed in a df and returns that df with an additional column, `Text`. 
       :param df: A dataframe of file information including a column for `File`
       :param file_directory: the main directory where files are stored
       :return: A dataframe with processed text '''

    # create copy to modify
    text_df = df.copy()

    # store processed text
    text = []

    # for each file (row) in the df, read in the file 
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        #print(filename)
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

            # standardize text using helper function
            file_text = process_file(file)
            # append processed text to list
            text.append(file_text)

    # add column to the copied dataframe
    text_df['Text'] = text

    return text_df


# In[ ]:


from unittest.mock import MagicMock, patch
import sklearn.naive_bayes
import numpy as np
import pandas as pd
import re

# test csv file
TEST_CSV = 'data/test_info.csv'

class AssertTest(object):
    '''Defines general test behavior.'''
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])

    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message

def _print_success_message():
    print('Tests Passed!')

# test clean_dataframe
def test_numerical_df(numerical_dataframe):

    # test result
    transformed_df = numerical_dataframe(TEST_CSV)

    # Check type is a DataFrame
    assert isinstance(transformed_df, pd.DataFrame), 'Returned type is {}.'.format(type(transformed_df))

    # check columns
    column_names = list(transformed_df)
    assert 'File' in column_names, 'No File column, found.'
    assert 'Task' in column_names, 'No Task column, found.'
    assert 'Category' in column_names, 'No Category column, found.'
    assert 'Class' in column_names, 'No Class column, found.'

    # check conversion values
    assert transformed_df.loc[0, 'Category'] == 1, '`heavy` plagiarism mapping test, failed.'
    assert transformed_df.loc[2, 'Category'] == 0, '`non` plagiarism mapping test, failed.'
    assert transformed_df.loc[30, 'Category'] == 3, '`cut` plagiarism mapping test, failed.'
    assert transformed_df.loc[5, 'Category'] == 2, '`light` plagiarism mapping test, failed.'
    assert transformed_df.loc[37, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'
    assert transformed_df.loc[41, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'

    _print_success_message()


def test_containment(complete_df, containment_fn):

    # check basic format and value 
    # for n = 1 and just the fifth file
    test_val = containment_fn(complete_df, 1, 'g0pA_taske.txt')

    assert isinstance(test_val, float), 'Returned type is {}.'.format(type(test_val))
    assert test_val<=1.0, 'It appears that the value is not normalized; expected a value <=1, got: '+str(test_val)

    # known vals for first few files
    filenames = ['g0pA_taska.txt', 'g0pA_taskb.txt', 'g0pA_taskc.txt', 'g0pA_taskd.txt']
    ngram_1 = [0.39814814814814814, 1.0, 0.86936936936936937, 0.5935828877005348]
    ngram_3 = [0.0093457943925233638, 0.96410256410256412, 0.61363636363636365, 0.15675675675675677]

    # results for comparison
    results_1gram = []
    results_3gram = []

    for i in range(4):
        val_1 = containment_fn(complete_df, 1, filenames[i])
        val_3 = containment_fn(complete_df, 3, filenames[i])
        results_1gram.append(val_1)
        results_3gram.append(val_3)

    # check correct results
    assert all(np.isclose(results_1gram, ngram_1, rtol=1e-04)),     'n=1 calculations are incorrect. Double check the intersection calculation.'
    # check correct results
    assert all(np.isclose(results_3gram, ngram_3, rtol=1e-04)),     'n=3 calculations are incorrect.'

    _print_success_message()

def test_lcs(df, lcs_word):

    test_index = 10 # file 10

    # get answer file text
    answer_text = df.loc[test_index, 'Text'] 

    # get text for orig file
    # find the associated task type (one character, a-e)
    task = df.loc[test_index, 'Task']
    # we know that source texts have Class = -1
    orig_rows = df[(df['Class'] == -1)]
    orig_row = orig_rows[(orig_rows['Task'] == task)]
    source_text = orig_row['Text'].values[0]

    # calculate LCS
    test_val = lcs_word(answer_text, source_text)

    # check type
    assert isinstance(test_val, float), 'Returned type is {}.'.format(type(test_val))
    assert test_val<=1.0, 'It appears that the value is not normalized; expected a value <=1, got: '+str(test_val)

    # known vals for first few files
    lcs_vals = [0.1917808219178082, 0.8207547169811321, 0.8464912280701754, 0.3160621761658031, 0.24257425742574257]

    # results for comparison
    results = []

    for i in range(5):
        # get answer and source text
        answer_text = df.loc[i, 'Text'] 
        task = df.loc[i, 'Task']
        # we know that source texts have Class = -1
        orig_rows = df[(df['Class'] == -1)]
        orig_row = orig_rows[(orig_rows['Task'] == task)]
        source_text = orig_row['Text'].values[0]
        # calc lcs
        val = lcs_word(answer_text, source_text)
        results.append(val)

    # check correct results
    assert all(np.isclose(results, lcs_vals, rtol=1e-05)), 'LCS calculations are incorrect.'

    _print_success_message()

def test_data_split(train_x, train_y, test_x, test_y):

    # check types
    assert isinstance(train_x, np.ndarray),        'train_x is not an array, instead got type: {}'.format(type(train_x))
    assert isinstance(train_y, np.ndarray),        'train_y is not an array, instead got type: {}'.format(type(train_y))
    assert isinstance(test_x, np.ndarray),        'test_x is not an array, instead got type: {}'.format(type(test_x))
    assert isinstance(test_y, np.ndarray),        'test_y is not an array, instead got type: {}'.format(type(test_y))

    # should hold all 95 submission files
    assert len(train_x) + len(test_x) == 95,         'Unexpected amount of train + test data. Expecting 95 answer text files, got ' +str(len(train_x) + len(test_x))
    assert len(test_x) > 1,         'Unexpected amount of test data. There should be multiple test files.'

    # check shape
    assert train_x.shape[1]==2,         'train_x should have as many columns as selected features, got: {}'.format(train_x.shape[1])
    assert len(train_y.shape)==1,         'train_y should be a 1D array, got shape: {}'.format(train_y.shape)

    _print_success_message()


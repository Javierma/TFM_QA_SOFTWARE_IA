import json
import os
import re
import csv
import subprocess
import sys

from screeninfo import get_monitors
import time
import selenium_tests
from getpass import getpass

import pandas as pd
import chardet


def generate_json(value):
    if value == 'Case':
        return '{"URL": "",\r\n "actions": [{"callFunction": "", "params": []}]}'

    else:
        return ''


def get_titles(types, tests):
    titles = []

    i = 0
    j = 0
    while i < len(types):
        if types[i] == 'Test' or types[i] == '*':
            titles.append(tests[i])
            j = i

        elif types[i] == 'Case':
            titles.append(tests[j])

        i = i + 1

    return titles


def get_screen_info():
    monitors = get_monitors()

    monitor_properties = None
    monitor_width = 0
    monitor_height = 0
    i = 0
    while i < len(monitors):
        if monitors[i].width > monitor_width and monitors[i].height > monitor_height:
            monitor_properties = monitors[i]

        i = i + 1

    return monitor_properties


def train_model():
    roberta_model = TFRobertaModel.frompretrained("roberta-base-uncased")
    roberta_tokenizer = RobertaTokenizer.frompretrained("roberta-base-uncased")

    data = tensorflow_datasets.load("")

    train_dataset = data["train"]
    validation_dataset = data["validation"]

    train_dataset = glue_convert_examples_to_features(train_dataset, roberta_tokenizer, 128, 'mrpc')
    validation_dataset = glue_convert_examples_to_features(validation_dataset, roberta_tokenizer, 128, 'mrpc')
    roberta_train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
    rooberta_validation_dataset = validation_dataset.batch(64)

    optimizer = tf.keras.optimizar.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    roberta_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    roberta_history = roberta_model.fit(
        roberta_train_dataset,
        epochs=2,
        steps_per_epoch=115,
        validation_data = roberta_validation_dataset,
        validation_steps=7
    )


def get_dxl():
    # Ask about the database, username, software and project
    database = '36677@NAVW1179.nav.es'
    username = 'jmarrieta'
    software = 'IMPACT'
    project_name = 'IMPACT V2'
    # database = input('Introduce la base de datos a la que debo conectarme: ')
    # username = input('Introduce tu nombre de usuario: ')
    # software = input('Introduce el nombre del software cuyos casos de prueba quieres extraer: ')
    # project_name = input('Introduce el nombre del proyecto/versión: ')

    onedrive_dir = os.getenv('OneDrive')
    # Save software and project_name in a file to be read by the DXL script, as the cannot be passed as parameters
    with open(onedrive_dir + '\Documents\DOORS_y_CHANGE\SCRIPTS DXL\project_sw_config.txt', mode='w') as f:
        f.write(software + '\n' + project_name)

    # Call the DXL script to get the test files
    command_to_execute = r'"C:\Program Files\IBM\Rational\DOORS\9.7\bin\doors.exe" -d ' + database + ' -user ' + username + ' -batch "' + onedrive_dir + '\Documents\DOORS_y_CHANGE\SCRIPTS DXL\TP_CSV_EXPORT.dxl" -a "\\' + database.split('@')[-1] + '\DXL\addins;\\' + database.split('@')[-1] + '\DXL" -J \\' + database.split('@')[-1] + '\DXL\project"'
    result = subprocess.call(command_to_execute)

    # Define a dictionary, which keys will be the unique words and their values will be the total amount
    word_counts = {}
    word_counts_lowercase = {}

    # Lists of all tests and titles along with the resulting dataframe
    result_df = pd.DataFrame(columns=['Type', 'Test', 'Title'])
    all_tests = []
    all_titles = []
    longest_entry_length = 0

    export_directory = 'C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_' + software + '\\'
    os.makedirs(export_directory, exist_ok=True)
    csv_filenames = os.listdir('C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_' + software + '\\')

    #Remove from list those files which are not CSV files
    csv_filenames = [csv_filename for csv_filename in csv_filenames if csv_filename.endswith('.csv')]

    # In case the resulting file is present, delete from csv_filenames list
    try:
        csv_filenames.remove(software + '_MODEL.csv')

    except ValueError:
        pass

    for csv_filename in csv_filenames:
        with open(export_directory + csv_filename, mode='rb') as f:
            encoding = chardet.detect(f.read())['encoding']

        filename = export_directory + csv_filename
        with open(filename, mode='r', encoding=encoding) as f:
            print(time.strftime('%d/%m/%Y %H:%M:%S') + '\tProcessing ' + filename + '...')
            file_content = f.read()
            file_content = file_content.split('\n')

            # Check if test case was tested for version V2.01.13 or lower. If higher, remove entries as they will not be tested
            i = 0
            while i < len(file_content):
                if 'Case' in file_content[i]:
                    j = i + 1
                    tested_versions = []
                    while j < len(file_content) and 'Case' not in file_content[j]:
                        if 'Execution' in file_content[j]:
                            tested_versions.append(file_content[j].split(';-;')[0])

                        j = j + 1

                    if '02.01.' not in ' '.join(tested_versions) and '02.00.' not in ' '.join(tested_versions):
                        file_content = file_content[0:i] + file_content[j:]

                i = i + 1

            # Remove lines containing Execution
            file_content = [content for content in file_content if ';-;Execution;-;' not in content]

            # Remove lines that are tests with no cases
            i = 0
            while i < len(file_content):
                if i < len(file_content) - 1 and ';Test;' in file_content[i] and ';Test;' in file_content[i + 1]:
                    file_content.remove(file_content[i])

                else:
                    i = i + 1

            # Check if the next line is part of a case, in which occurrence we add a \n and write the value between double commas
            i = 2
            while i < len(file_content):
                if i < (len(file_content) - 1) and 'Case' not in file_content[i + 1] and 'Test' not in file_content[i + 1] and '*;' not in file_content[i + 1]:
                    file_content[i] = file_content[i] + '\r\n' + file_content[i + 1]
                    file_content.remove(file_content[i + 1])

                else:
                    i = i + 1

            # Now separate the columns
            i = 0
            while i < len(file_content):
                file_content[i] = file_content[i].split(';-;')
                if len(file_content[i]) > 1:
                    file_content[i] = file_content[i][2:]

                i = i + 1

        df = pd.DataFrame(columns=file_content[0], data=file_content[1:])

        # Get the column that has all values and get all words and their total amount
        values = df['Test'].tolist()
        i = 0
        while i < len(values):
            words = re.findall(r'(\b[a-zA-Z0-9]+\b)', values[i])
            for word in words:
                try:
                    word_counts[word] = word_counts[word] + 1
                    word_counts_lowercase[word.lower()] = word_counts_lowercase[word.lower()] + 1

                except KeyError:
                    word_counts[word] = 1
                    word_counts_lowercase[word.lower()] = 1

            i = i + 1

        # Add column Title, which whill be the object heading. The object heading is the "title of case" or text in bold in doors
        df['Title'] = get_titles(df.Type, df.Test)

        result_df = pd.concat([result_df, df], ignore_index=True)
        l = 0

    new_cols_order = ['Type', 'Title', 'Test']
    result_df = result_df[new_cols_order]

    # Add new column with a default JSON so as to ease edition
    result_df['JSON'] = result_df['Type'].apply(generate_json)

    final_csv = export_directory + software + '_MODEL.csv'
    print(time.strftime('%d/%m/%Y %H:%M:%S') + '\tSaving file ' + final_csv)

    continue_trying = True
    while continue_trying:
        try:
            result_df.to_csv(final_csv, index=None, sep=';', encoding='windows-1252')
            continue_trying = False

        except PermissionError:
            print('Error when trying to save ' + final_csv + '! Close file in Excel so as to save the file')
            time.sleep(5)

    # Get the longest test length and also the number of attributes with string is longer than 512 words
    values = result_df['Test'].values.tolist()
    i = 0
    longest_entry_index = 0
    num_entries_longer_512 = 0
    while i < len(values):
        text_entry = values[i].split()

        # Remove punctuation signs, as they would not count as tokens
        while '.' in text_entry:
            text_entry.remove(text_entry[text_entry.index('.')])

        while ',' in text_entry:
            text_entry.remove(text_entry[text_entry.index(',')])

        while ';' in text_entry:
            text_entry.remove(text_entry[text_entry.index(';')])

        while '_' in text_entry:
            text_entry.remove(text_entry[text_entry.index('_')])

        while '-' in text_entry:
            text_entry.remove(text_entry[text_entry.index('-')])

        while '"' in text_entry:
            text_entry.remove(text_entry[text_entry.index('"')])

        while '\'' in text_entry:
            text_entry.remove(text_entry[text_entry.index('\'')])

        if len(text_entry) > longest_entry_length:
            longest_entry_length = len(values[i].split())
            longest_entry_index = i

        if len(text_entry) > 512:
            num_entries_longer_512 = num_entries_longer_512 + 1

        i = i + 1

    print('Total entries, including titles:', len(values))
    print('\nUnique words: ' + str(len(word_counts.keys())))
    print('\nUnique words with all characters in lowercase: ' + str(len(word_counts_lowercase.keys())))
    print('\nIndexes longer than 512 words:', num_entries_longer_512,
          '\nLongest length of a text entry (in words, including punctuation signs):', longest_entry_length, '.Index:',
          longest_entry_index, '\nText:', values[longest_entry_index])

    with open('word_counts.csv', mode='w', encoding='windows-1252') as f:
        f.write('Word;Times written\n')
        for word_key in word_counts.keys():
            f.write(word_key + ';' + str(word_counts[word_key]) + '\n')

    with open('word_counts_lowercase.csv', mode='w', encoding='windows-1252') as f:
        f.write('Word;Times written\n')
        for word_key in word_counts_lowercase.keys():
            f.write(word_key + ';' + str(word_counts_lowercase[word_key]) + '\n')
    l = 0


def execute_tests():
    screen_info = get_screen_info()
    my_tests = selenium_tests.Tests(screen_info)
    my_tests.setup_method()
    json_str = '{"URL": "https://impact-neo-ced.enaire.es/#/login?redirect=%2Fmain%2Fmaster", "actions": [{"callFunction": "check_image_present", "params": [{"alt": "logo enaire"}]}, {"callFunction": "check_existing_inputs", "params": [{"id": "input-16", "text": ""}, {"id": "password", "text": ""}]}, {"callFunction": "check_elements_exist", "params": [{"type": "button", "text": "ACCESS PROBLEM"}, {"type": "button", "text": "Login"}]}]}'
    json_decoder = json.JSONDecoder()
    json_dict = json_decoder.decode(json_str)

    # my_tests.test_login()

    for action in json_dict['actions']:
        match action['callFunction']:
            case 'check_elements_exist':
                my_tests.go_to_url(json_dict['URL'])
                my_tests.check_elements_exist(action['params'])

    input('Press Enter when ready to exit')
    my_tests.teardown_method()


def main():
    if len(sys.argv) > 0:
        if '-h' in sys.argv or '--help' in sys.argv:
            print('Options:\n -h,--help\tShow program options and examples\n-d,--get-dxl\tGet DOORS Tests DXL content and export to CSV file\n-t,--tests\tExecute tests')

        elif '-d' in sys.argv or '--get-dxl' in sys.argv:
            get_dxl()

        elif '-t' in sys.argv or '--tests' in sys.argv:
            execute_tests()

    exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


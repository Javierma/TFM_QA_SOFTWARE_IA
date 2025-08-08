import json
import os
import re
import csv
import pickle
import numpy
import yaml
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

def generar_yaml(value):
    if value == 'Case':
        return 'URL: \r\n acciones: \r\n  - funcion:\r\n    params: '

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

def obtener_cabecera_test(test):
    return test.split('\r\n')[0]

def obtener_dxl():
    database = '36677@NAVW1179.nav.es'
    username = 'jmarrieta'
    software = 'IMPACT'
    project_name = 'IMPACT V2'
    '''
    # Ask about the database, username, software and project
    database = input('Introduce la base de datos a la que debo conectarme: ')
    username = input('Introduce tu nombre de usuario: ')
    software = input('Introduce el nombre del software cuyos casos de prueba quieres extraer: ')
    project_name = input('Introduce el nombre del proyecto/versión: ')

    onedrive_dir = os.getenv('OneDrive')
    # Save software and project_name in a file to be read by the DXL script, as they cannot be passed as parameters
    with open(onedrive_dir + '\Documents\DOORS_y_CHANGE\SCRIPTS DXL\project_sw_config.txt', mode='w') as f:
        f.write(software + '\n' + project_name)

    # Call the DXL script to get the test files
    command_to_execute = r'"C:\Program Files\IBM\Rational\DOORS\9.7\bin\doors.exe" -d ' + database + ' -user ' + username + ' -batch "' + onedrive_dir + '\Documents\DOORS_y_CHANGE\SCRIPTS DXL\TP_CSV_EXPORT.dxl" -a "\\' + database.split('@')[-1] + '\DXL\addins;\\' + database.split('@')[-1] + '\DXL" -J \\' + database.split('@')[-1] + '\DXL\project"'
    result = subprocess.call(command_to_execute)
    '''
    # Define a dictionary, which keys will be the unique words and their values will be the total amount
    word_counts = {}
    word_counts_lowercase = {}

    # Lists of all tests and titles along with the resulting dataframe
    df_resultante = pd.DataFrame(columns=['Tipo', 'Test', 'Titulo'])
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
        csv_filenames.remove(software + '_MODEL_copia.csv')

    except ValueError:
        pass

    try:
        csv_filenames.remove(software + '_MODEL_copia.csv')

    except ValueError:
        pass

    for csv_filename in csv_filenames:
        with open(export_directory + csv_filename, mode='rb') as f:
            encoding = chardet.detect(f.read())['encoding']

        filename = export_directory + csv_filename
        with open(filename, mode='r', encoding=encoding) as f:
            print(time.strftime('%d/%m/%Y %H:%M:%S') + '\tProcesando ' + filename + '...')
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

        # Renombramos las columnas
        df.columns = ['Tipo', 'Test']

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
        df['Titulo'] = get_titles(df.Tipo, df.Test)

        df_resultante = pd.concat([df_resultante, df], ignore_index=True)
        l = 0

    new_cols_order = ['Tipo', 'Titulo', 'Test']
    df_resultante = df_resultante[new_cols_order]

    # Hay casos de pruebas duplicados, donde una entrada hace referencia a una versión anterior a la que se está probando. Las buscamos y eliminamos
    df_resultante['Cabecera caso'] = df_resultante['Test'].apply(obtener_cabecera_test)

    # Comprobamos aquellos casos donde la cabecera es la misma a excepción de que una indica la versión y otra no
    cabeceras_sin_version = [cabecera for cabecera in df_resultante['Cabecera caso'] if '(V2' not in cabecera]

    for cabecera in cabeceras_sin_version:
        # Buscamos las filas que contienen cabecera, incuyendo la versión a partir de la cual aplica
        filas_encontradas = df_resultante.loc[(df_resultante['Test'].str.contains(cabecera + ' (V2', regex=False)) | (df_resultante['Cabecera caso'].str.contains(cabecera + '(V2', regex=False)) | (df_resultante['Cabecera caso'].str.contains(cabecera + '(v2', regex=False)) | (df_resultante['Cabecera caso'].str.contains(cabecera + '(v2', regex=False))]

        if len(filas_encontradas) > 0:
            # Obtenemos el contenido del test sin la cabecera, que se usará para comprobar que la única diferencia es la versión
            texto_test = '\r\n'.join(filas_encontradas.loc[filas_encontradas.index[0]]['Test'].split('\r\n')[1:])

            # Buscamos las filas que contienen la cabecera que estamos buscando, sin la versión, y obtenemos el índice
            filas_originales = df_resultante.loc[df_resultante['Test'].str.contains(cabecera + '\r\n', regex=False)]
            indice = filas_originales.index[0]

            # Si, en efecto, se trata de filas idénticas salvo la versión en la cabecera, borramos la entrada anterior
            if texto_test in filas_originales.loc[indice]['Test']:
                df_resultante.drop(index=indice, inplace=True)


    # Eliminamos la columna dado que ya no resulta necesaria
    df_resultante.drop(columns=['Cabecera caso'], inplace=True)

    # Reordenamos las columnas por el contenido de Test. De esa forma, se facilitará encontrar filas que puedan ser idénticas
    df_resultante.sort_values(by=['Titulo', 'Test'], inplace=True)

    # Ordenamos por el contenido en test
    df_resultante.reset_index(inplace=True, drop=True)

    # Add new column with a default YAML so as to ease edition
    df_resultante['YAML'] = df_resultante['Tipo'].apply(generar_yaml)

    final_csv = export_directory + software + '_MODEL.csv'
    print(time.strftime('%d/%m/%Y %H:%M:%S') + '\tGuardando archivo ' + final_csv)

    continue_trying = True
    while continue_trying:
        try:
            df_resultante.to_csv(final_csv, index=None, sep=';', encoding='windows-1252')
            continue_trying = False

        except PermissionError:
            print('Error when trying to save ' + final_csv + '! Close file in Excel so as to save the file')
            time.sleep(5)

    # Get the longest test length and also the number of attributes with string is longer than 512 words
    values = df_resultante['Test'].values.tolist()
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


def obtener_diccionario_json(texto_json):
    texto_json = texto_json.replace('{"URL": "https://impact-neo-ced.enaire.es/#/main/master"', 'URL: https://impact-neo-ced.enaire.es/#/main/master')
    texto_json = texto_json.replace('acciones: [{"funcion: "", "params"": {}}]', 'acciones\r\n  - funcion: \r\n   params: ')

    json_dict = None
    try:
        json_decoder = json.JSONDecoder()
        json_dict = json_decoder.decode(texto_json)

    except json.decoder.JSONDecodeError:
        return ''
        pass

        # El texto JSON contiene errores. Intentamos detectar a qué se debe y sugerimos corrección
        # Revisamos en primer lugar el inicio y fin de la cadena que contiene el texto en JSON
        if texto_json.startswith('{') and texto_json.endswith('}') is False and texto_json.endswith(']') is False:
            texto_json = texto_json + '}'

        elif texto_json.startswith('{') is False and texto_json.startswith('[') is False and texto_json.endswith('}'):
            texto_json = '{' + texto_json

        elif texto_json.startswith('[') and texto_json.startswith('{') is False and texto_json.endswith(']') is False:
            texto_json = texto_json + ']'

        elif texto_json.startswith('[') is False and texto_json.endswith(']'):
            texto_json = '[' + texto_json

        continuar_revisando = True
        while continuar_revisando:
            # Eliminamos espacios innecesarios que puedan dificultar la detección de errores
            texto_json = re.sub(r'(\s{0,}\:\s{0,})', ':', texto_json)
            texto_json = re.sub(r'(\s{0,}\,\s{0,})', ',', texto_json)
            texto_json = re.sub(r'(\s{0,}\{\s{0,})', '{', texto_json)
            texto_json = re.sub(r'(\s{0,}\}\s{0,})', '}', texto_json)
            texto_json = re.sub(r'(\s{0,}\[\s{0,})', '[', texto_json)
            texto_json = re.sub(r'(\s{0,}\]\s{0,})', ']', texto_json)

            # Empezamos por las dobles comillas
            # El texto que va tras un = no tiene " ni al inicio ni al fin
            resultados_encontrados = re.findall(r'(=\\\\^["].+\b\\\\^["])', texto_json)
            if len(resultados_encontrados) > 0:
                for resultado in resultados_encontrados:
                    texto_json = texto_json.replace(resultado, resultado[:3] + '"' + resultado[3:] + '"')

            # El texto que va tras un = no tiene \ ni al inicio ni al fin
            resultados_encontrados = re.findall(r'(=".+\b")', texto_json)
            if len(resultados_encontrados) > 0:
                for resultado in resultados_encontrados:
                    texto_json = texto_json.replace(resultado, resultado[0] + '\\' + resultado[1:-1] + '\\' + resultado[-1])

            # El texto que va tras un = no tiene " al inicio
            resultados_encontrados = re.findall(r'(=\\\\^["].+\b)', texto_json)
            if len(resultados_encontrados) > 0:
                for resultado in resultados_encontrados:
                    texto_json = texto_json.replace(resultado, resultado[:3] + '"' + resultado[1:])

            # El texto que va tras un = no tiene " al final
            resultados_encontrados = re.findall(r'(=\\\\".+\b\\\\^["])', texto_json)
            if len(resultados_encontrados) > 0:
                for resultado in resultados_encontrados:
                    texto_json = texto_json.replace(resultado, resultado + '"')

            # El texto no tiene " tras , o : o [ o { o (
            resultados_encontrados = re.findall(r'((?:\{|\[|\:|\,)[a-zA-Z]+)', texto_json)
            if len(resultados_encontrados) > 0:
                for resultado in resultados_encontrados:
                    texto_json = texto_json.replace(resultado, resultado[0] + '"' + resultado[1:])

            # El texto no tiene " tras , o : o ] o } o )
            resultados_encontrados = re.findall(r'(\"\w+(?:\}|\]|\:|\,))', texto_json)
            if len(resultados_encontrados) > 0:
                for resultado in resultados_encontrados:
                    texto_json = texto_json.replace(resultado, resultado[:-1] + '"' + resultado[-1])

            # } y { deben is separados por coma y no por punto. En caso de detectar algún caso se cambiará
            texto_json = re.sub(r'(\}\.\{)', '},{', texto_json)
            texto_json = re.sub(r'(\}\{)', '},{', texto_json)
            continuar_revisando = False
            '''
            caracteres_apertura_cierra = []
            i = 0
            while i < len(texto_json):
                if texto_json[i] == '(' or texto_json[i] == '{' or texto_json[i] == '[' or texto_json[i] == ')' or texto_json[i] == '}' or texto_json[i] == ']':
                    caracteres_apertura_cierra.append((i, texto_json[i]))

                i = i + 1

            i = 0
            while i < len(caracteres_apertura_cierra):
                # Empezamos desde dentro hacia fuera. Para ello, buscamos el carácter de apertura que se encuentra más adentro
                while caracteres_apertura_cierra[i][1] == '(' or caracteres_apertura_cierra[i][1] == '{' or caracteres_apertura_cierra[i][1] == '[':
                    i = i + 1

                i = i - 1
                j = caracteres_apertura_cierra[i][0]
                k = caracteres_apertura_cierra[i][0]
                match caracteres_apertura_cierra[i][1]:
                    # Buscamos el primer carácter de cierre que haya en cada caso
                    case '(':
                        while texto_json[j] != ')' and texto_json[j] != '}' and texto_json[j] != ']':
                            j = j + 1
                            
                        if texto_json[j] == ')':
                            caracteres_apertura_cierra.remove((j,')'))
                            caracteres_apertura_cierra.remove(caracteres_apertura_cierra[i])

                        else:
                            parte_texto = texto_json[k:j+1]
                            texto_json = re.sub(r'(\(.+[^\)]\])', parte_texto, parte_texto.replace(']"', ')]"'))

                    case '{':
                        while texto_json[j] != '}' and texto_json[j] != ')' and texto_json[j] != ']':
                            j = j + 1

                        if texto_json[j] == '}':
                            caracteres_apertura_cierra.remove((j,'}'))
                            caracteres_apertura_cierra.remove(caracteres_apertura_cierra[i])

                        else:
                            parte_texto = texto_json[k:j+1]
                            texto_json = re.sub(r'(\{.+[^}],{)', parte_texto, parte_texto.replace(',{', '},{'))

                    case '[':
                        while texto_json[j] != ']' and texto_json[j] != '}' and texto_json[j] != ')':
                            j = j + 1

                        if texto_json[j] == ']':
                            caracteres_apertura_cierra.remove((j,']'))
                            caracteres_apertura_cierra.remove(caracteres_apertura_cierra[i])

                        else:
                            parte_texto = texto_json[k:j+1]
                            texto_json = re.sub(r'(^\[.+""})', parte_texto, parte_texto.replace('""', '"]"'))
                            l = 0

                i = i + 1

            l = 0

            # Contamos la cantidad de caracteres {, }, [, ], ", ( y ). Si para el carácter de apertura, como por ejemplo {, no existe el mismo número de } que de {, se ha de seguir revisando
            contador_comillas_dobles = texto_json.count('"')
            contador_llave_apertura = texto_json.count('{')
            contador_llave_cierre = texto_json.count('}')
            contador_corchete_apertura = texto_json.count('[')
            contador_corchete_cierre = texto_json.count(']')
            contador_parentesis_apertura = texto_json.count('(')
            contador_parentesis_cierre = texto_json.count(')')

            if (contador_comillas_dobles%2) == 0 and contador_llave_apertura == contador_llave_cierre and contador_parentesis_apertura == contador_parentesis_cierre and contador_corchete_apertura == contador_corchete_cierre:
                continuar_revisando = False
                
        '''

    try:
        json_decoder = json.JSONDecoder()
        json_dict = json_decoder.decode(texto_json)

    except json.decoder.JSONDecodeError:
        return None

    return json_dict


def procesar_json(tests_obj, texto_json):
    # Procesamos el JSON y lo convertimos a diccionario
    diccionario_json = obtener_diccionario_json(texto_json)

    # Obtenemos el listado de acciones a realizar
    acciones = diccionario_json['acciones']

    # Ejecutamos cada una de las acciones. La función getattr nos permitirá poder llamar a la función que corresponde teniendo la función como string
    for accion in acciones:
        funcion = getattr(tests_obj, accion['funcion'])

        # Es posible que se requiera realizar una acción varias veces. Para ello, comprobamos si una de las claves del diccionario es veces
        if 'veces' not in accion['params']:
            funcion(accion['params'])

        else:
            i = 0
            while i < accion['veces']:
                funcion(accion['params'])
                i = i + 1


def convert_to_yaml(row):
    if row is not numpy.nan:
        json_dict = obtener_diccionario_json(row)
        yaml_text = yaml.dump(json_dict, Dumper=yaml.Dumper)
        return yaml_text

    else:
        return None

def execute_tests():
    with open('df_xpaths.pickle', mode='rb') as fichero_pickle:
        df_xpaths = pickle.load(fichero_pickle)

    l = 0

    '''
    detected_encoding = None
    with open('C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_IMPACT\\IMPACT_MODEL.csv', mode='rb') as f:
        file_content = f.read()
        detected_encoding = chardet.detect(file_content)['encoding']

    f.close()

    modelo_csv = pd.read_csv('C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_IMPACT\\IMPACT_MODEL.csv', sep=';', encoding=detected_encoding)

    contenido_yaml = yaml.safe_load(modelo_csv.loc[1483]['YAML'])
    '''
    l = 0
    '''
    detected_encoding = None
    with open('C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_IMPACT\\IMPACT_MODEL_copia.csv', mode='rb') as f:
        file_content = f.read()
        detected_encoding = chardet.detect(file_content)['encoding']

    f.close()

    modelo_csv = pd.read_csv('C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_IMPACT\\IMPACT_MODEL_copia.csv', sep=';', encoding='windows-1252')
    modelo_csv['YAML'] = modelo_csv['JSON'].apply(convert_to_yaml)
    modelo_csv.to_csv('C:\\Users\\jmarrieta\\OneDrive - ENAIRE\\Desktop\\TPs_IMPACT\\IMPACT_MODEL_copia.csv', sep=';', encoding=detected_encoding)
    l = 0
    '''
    screen_info = get_screen_info()
    my_tests = selenium_tests.Tests(screen_info)
    my_tests.guardar_df_como_excel(df_xpaths, 'xpaths.xlsx')
    exit(0)
    #my_tests.setup_method()

    params = {'centro':'GCCC'}
    params = {}
    my_tests.iniciar_sesion(params)

    # Esperamos a que el usuario indique que se puede continuar
    input('Presione Enter cuando esté listo para la obtención de xpaths')

    df_xpaths = pd.DataFrame(columns=['XPATH', 'PNG', 'ACCESO DESDE', 'PALABRAS CLAVE', 'CONTENIDO HTML', 'XPATHS_CONTENIDO_HTML', 'REVISADO'])

    print ('Indique el tipo de obtención de Xpaths deseado:\n1) Voraz\n2) Asistido')
    tipo_obtencion = input(': ')

    parado = None
    if tipo_obtencion == '1':
        parado = my_tests.obtener_xpaths_voraz(df_xpaths)

    elif tipo_obtencion == '2':
        parado = my_tests.obtener_xpaths_asistido(df_xpaths)

    else:
        print('Valor desconocido')

    '''
    try:
        my_tests.obtener_xpaths(df_xpaths)

    except KeyboardInterrupt:
        # Guardamos fichero pickle para, en caso de volver a ejecutar, poder hacerlo desde donde nos hubiésemos quedado
        pickle.dump(df_xpaths, 'df_xpaths.pickle')

        # Acciones como CTRL+C provocan este error. Lo capturamos para poder guardar lo hecho y por si se quiere revisar durante el proceso que va por buen camino
        print('Se ha detectado la ejecución de CTRL+C. Guardando hasta lo obtenido')
        my_tests.guardar_df_como_excel(df_xpaths, 'xpaths.xlsx')
    '''
    #my_tests.comprobar_contenido_hover(params)
    #my_tests.comprobar_resolucion_pantalla(params)
    #my_tests.comprobar_version_driver(params)
    #time.sleep(30)
    #my_tests.poner_quitar_pantalla_completa(params)
    if parado is None:
        my_tests.cerrar_sesion()

    '''
    params = {'login_type': 'empty_fields'}
    my_tests.test_login(params)

    params = {'login_type': 'empty_username'}
    my_tests.test_login(params)

    params = {'login_type': 'empty_password'}
    my_tests.test_login(params)

    params = {'login_type': 'wrong_input'}
    my_tests.test_login(params)
    '''
    input('Press Enter when ready to exit')
    my_tests.teardown_method()

    #my_tests.check_cerrar_sesion()


def main():
    dct = yaml.safe_load('''
    name: John
    age: 30
    automobiles:
    - brand: Honda
      type: Odyssey
      year: 2018
    - brand: Toyota
      type: Sienna
      year: 2015
    ''')

    dct = yaml.safe_load('''
    URL: https://impact-neo-ced.enaire.es/#/main/master
    acciones:
    - funcion: comprobar_orden_elementos
      params:
        inicio: izquierda
        fin: derecha
        elementos:
        - xpath: //img[@alt="logo enaire"]
        - xpath: //div[contains(@class,"work-space")]
        - xpath: //div[contains(@class,"simulationMngtClass")]
        - xpath: //div[contains(@class,"aixm v-card")]     
        - xpath: //div[contains(@class,"current-time")]
        - xpath: //div[contains(@id,"user")]
    ''')

    if len(sys.argv) > 0:
        if '-h' in sys.argv or '--help' in sys.argv:
            print('Options:\n -h,--help\tShow program options and examples\n-d,--get-dxl\tGet DOORS Tests DXL content and export to CSV file\n-t,--tests\tExecute tests')

        elif '-d' in sys.argv or '--get-dxl' in sys.argv:
            obtener_dxl()

        elif '-t' in sys.argv or '--tests' in sys.argv:
            execute_tests()

    exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


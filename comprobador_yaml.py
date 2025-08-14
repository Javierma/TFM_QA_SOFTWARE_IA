import yaml

dct = yaml.safe_load('''
  - funcion: guardar_valor
    params: 
      - xpath: //div[contains(@class, "modal-title")]
        id: id_vuelo_2
        valor_a_guardar:
        - regex: \\bw+\\b
    ''')

print(dct)
l=0
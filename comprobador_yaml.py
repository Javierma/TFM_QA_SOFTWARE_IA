import yaml

dct = yaml.safe_load('''
URL: https://impact-neo-ced.enaire.es/#/main/master
acciones:
  - funcion: hacer_clic
    params: 
      - xpath: //div[contains(@class,"tab-ws")]/div/span[contains(text(),"OPER") or contains(text(),"FCST") or contains(text(),"SIMU")]
      
  - funcion: hacer_clic
    params: 
      - xpath: //div[@id="leftPanel"]//*[name()="path"][contains(@d, "M12,15.5A3.5")]
      
  - funcion: rellenar_campo
    params: 
      - xpath: //input[contains(@placeholder,"Identifier...")]
        valor: 
          - xpath: //div[contains(@class, "nom-graph UXUIbigger")]
          
  - funcion: presionar_tecla
    params:
      - xpath: //input[contains(@placeholder,"Identifier...")]
        tecla: Enter
    ''')

print(dct)
l=0
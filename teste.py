import BBFinance as bb

lista = [] 
ativos = ['PETR4.SA', 'VALE3.SA', 'BBDC4.SA', 'BBAS3.SA', 'ITUB4.SA', 'ITSA4.SA', 'ABEV3.SA', 
                     'WEGE3.SA', 'B3SA3.SA', 'MGLU3.SA', 'GGBR4.SA', 'CSNA3.SA', 'ENBR3.SA', 
                     'COGN3.SA'] 

for i in ativos:
    print(i)
    spot = bb.get_info(symbol= i) 
    lista.append(spot)


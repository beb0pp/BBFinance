#BIBLIOTECAS PARA ANALISE DE ATIVOS
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

#BIBLIOTECAS PARA WEBSCRAPING
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller
import requests
from bs4 import BeautifulSoup

#BIBLIOTECAS DE CHAMADAS DE API
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import uvicorn

#BIBLIOTECAS DE ANALISE DE DATAFRAMES, DADOS E CRIAÇAO DE CLASSES
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel
from typing import List, Union, Optional

import warnings
warnings.filterwarnings("ignore")

# chromedriver_autoinstaller.install()
yf.pdr_override()

today = datetime.today()
# Data de um ano atrás
one_year_ago = today - timedelta(days=365)

# Convertendo as datas para strings com formato yyyy-mm-dd
oneY = one_year_ago.strftime('%Y-%m-%d')
currently = today.strftime('%Y-%m-%d')

app = FastAPI()
templates = Jinja2Templates(directory="Site")
# app.mount("/Static", StaticFiles(directory="Static"), name="Static")

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


class TipoSetores(str, Enum):
    Corporativas = "Lajes Corporativas"
    Mobiliarios = "Títulos e Val. Mob."
    Shoppings = "Shoppings"
    Hibridos = 'Híbrido'
    Renda = 'Renda'
    Logistica = 'Logística'
    Hospital = 'Hospital'
    Residencial = 'Residencial'
    Outros = 'Outros'

class TipoPerfis(str, Enum):
    Agressivo = 'Agressivo'
    Moderado = 'Moderado'
    Conservador = 'Conservador'

def formataValoresNumero(df, nomeColuna):
    df[nomeColuna] = df[nomeColuna].replace('[.]', '', regex=True)
    df[nomeColuna] = df[nomeColuna].replace('[%]', '', regex=True)
    df[nomeColuna] = df[nomeColuna].replace('[,]', '.', regex=True)
    df[nomeColuna] = df[nomeColuna].astype(float)

    return df

## INFOS DAS AÇOES ##
@app.get("/stocks/{symbol}/info", response_model=None)
def get_info(symbol: str) -> dict:
    
    """
    ## Usabilidade 
    - Busca as principais informaçoes sobre o ativo selecionado como Preço e Dividendos \n
    
    ## Parâmetros
    - symbol -> Nome do Ativo para a busca \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO
        tipoInfo = type(info)
        if tipoInfo == dict:
            pass #SE FOR VAI SO PASSAR 
    except:
        return {"error": print("Invalid ticker symbol")}
    
    # Obtenha o valor a mercado da açao
    current_price = stock.info['regularMarketPrice']

    # Obtenha o nome completo da empresa
    company_name = stock.info['longName']

    #Valor de Dividendos
    dividend = stock.dividends
    dividend = dividend.iloc[-1:].sum()
    
    # Crie um objeto JSON com as informações da ação
    json_data = {'symbol': symbol, 
                 'current_price': current_price, 
                 'company_name': company_name,
                 'dividends' : dividend}
    
    formatted_json = json.dumps(json_data, indent=5)
    print(formatted_json)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/info")
response = Response(media_type="application/json")


## HISTORICO DAS AÇOES ##

@app.get("/stocks/{symbol}/history", response_model=None)
def get_stock_history(symbol: str, period: str = '1y') -> pd.DataFrame:
    
    """
    ## Usabilidade 
    - Usada para verificar o histórico da açao selecionada e em qual periodo \n
    
    ## Parâmetros
    
    - symbol -> Nome do Ativo para a busca \n
    - period -> Data em ANOS para a busca das informaçoes do Ativo \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO
        tipoInfo = type(info)
        if tipoInfo == dict:
            pass #SE FOR VAI SO PASSAR 
    except:
        return {"error": print("Invalid ticker symbol")}
    
    history = stock.history(period=period)
    
    if history.empty:
        return {"error": print("No data found")}
    else:
        history_dict = history.to_dict(orient="list")
        history_df = pd.DataFrame.from_dict(history_dict).reset_index(drop=False)     
        return print(history_df)
        # json_data = {'symbol': symbol,
        # "history":  history_df.to_dict(orient="records"),
        # }

        # formatted_json = json.dumps(json_data, indent=2)
        # print(formatted_json)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/history")
responseHistory = Response(media_type="application/json")


## TENDENCIA DE PREÇO ##

@app.get("/stock/{symbol}/trend", response_model=None)
def get_stock_trend(symbol: str) -> dict:
    
    """
    ## Usabilidade 
    - Identifica a tendencia de preço de uma açao, se ira ser de ALTA ou BAIXA
    
    ## Parâmetros
    
    - symbol -> Nome do Ativo para a busca \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO
        tipoInfo = type(info)
        if tipoInfo == dict:
            pass #SE FOR VAI SO PASSAR 
    except:
        return {"error": print("Invalid ticker symbol")}

    history = stock.history(period='1d')
    close_prices = history['Close']
    trend = 'up' if close_prices.iloc[-1] > close_prices.iloc[0] else 'down'
    
    json_data = { "symbol": symbol,
                "trend": trend,
    }
    
    formatted_json = json.dumps(json_data, indent=2)
    print(formatted_json)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/trend")
responseHistory = Response(media_type="application/json")


## RSI ##

@app.get("/stock/{symbol}/technical", response_model=None)
def get_stock_technicals(symbol: str) -> dict:
    
    """
    ## Usabilidade 
    - cálculo envolve a comparação da média de ganhos em um período de tempo com a média de perdas em um período de tempo. \n
    
    ## Como interpretar 
    - Quando o RSI está acima de 70, o ativo é considerado sobrecomprado, o que significa que pode estar prestes a sofrer uma correção para baixo. 
    Quando o RSI está abaixo de 30, o ativo é considerado sobrevendido, o que significa que pode estar prestes a subir novamente. \n 
    
    ## Parâmetros
    - symbol -> Nome do Ativo para a busca \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO, SENAO VAI PASSAR
        tipoInfo = type(info)
        if tipoInfo == dict:
            pass 
    except:
        return {"error": print("Invalid ticker symbol")}

    history = stock.history(period='max')
    close_prices = history['Close']
    
    # Calcula as médias móveis
    sma_50 = close_prices.rolling(window=50).mean().iloc[-1] #calcula as médias móveis de 50 períodos
    sma_200 = close_prices.rolling(window=200).mean().iloc[-1]  #calcula as médias móveis de 200 períodos
    
    # Calcula o Índice de Força Relativa (RSI)
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    if rsi >= 70:
        status = 'A chance do preco do ativo CAIR'
    else:
        status = 'A chance do preco do ativo SUBIR'
    
    json_data = {
        "symbol": symbol,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi": rsi,
        "tendency" : status
    }

    formatted_json = json.dumps(json_data, indent=4)
    print(formatted_json)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/technical")
responseHistory = Response(media_type="application/json")


## VOLATILIDADE ##

@app.get("stocks/{symbol}/volatility", response_model=None)
def get_volatility(symbol: str, start_date: str, end_date: str) -> str:
    
    """
    ## Usabilidade 
    - Método usado para verificar a volatilidade de um ativo em comparacao ao mercado em que esta  \n
    
    ## Parâmetros
    - symbol -> Nome do Ativo para a busca \n
    - start_date -> Data de Inicio da busca das infos (preco, volume, etc) do ativo \n
    - end_date -> Data Final para a busca das infos (preco, volume, etc) do ativo \n
    """
    
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            return {"error": print("Nao foi encontrado o historico nesse periodo, verificar.")}
    except:
        pass
    
    log_returns = np.log(stock_data['Close']/stock_data['Close'].shift(1))
    volatility = np.sqrt(252*log_returns.var())
    return {'volatility': print(volatility)}

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/volatility")
responseHistory = Response(media_type="application/json")


## BETA ##

@app.get("stocks/{symbol}/beta", response_model=None)
def get_beta(symbol: str) -> dict:
    
    """
    ## Usabilidade 
    - O beta é uma medida estatística que indica a relação entre a volatilidade de uma ação e a volatilidade do mercado como um todo.
    O valor do beta é utilizado para medir o risco de uma ação em relação ao mercado em que ela é negociada. \n
    
    ## Parâmetros
    - symbol -> Nome do Ativo para a busca \n
    - market -> Como padrao, Mercado: IBOVESPA / BVSP
    """
    
    # Obter os dados do ativo e do mercado
    try:
        asset = yf.Ticker(symbol)
        market = yf.Ticker("^BVSP") # Índice Bovespa como mercado de referência
        info = asset.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO, SENAO VAI PASSAR
        infoMarket = market.info
        tipoInfo = type(info)
        if tipoInfo == dict | infoMarket == dict:
            pass 
    except:
        return {"error": print("Invalid ticker symbol")}

    asset_history = asset.history(period="max")
    market_history = market.history(period="max")

    # Calcular os retornos diários
    asset_returns = asset_history['Close'].pct_change()
    market_returns = market_history['Close'].pct_change()

    # Calcular o beta
    cov = asset_returns.cov(market_returns)
    var = market_returns.var()
    beta = cov / var

    if beta > 1:
        status = 'Acao mais Volatil que o mercado em geral'
    if beta < 1:
        status = 'Acao menos Volatil que o mercado em geral'
    if beta == 1:
        status =  'Acao com a mesma Volatilidade que o mercado em geral'


    json_data =  {"beta": beta,
                  "status" : status}
    
    formatted_json = json.dumps(json_data, indent=2)
    print(formatted_json)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/beta")
responseHistory = Response(media_type="application/json")


## VAR ##
    
@app.get("stocks/{symbol}/VaR", response_model=None)
def get_var(symbol: str, confidence_level: float, lookback_period: int) -> dict:
    
    """
    ## Usabilidade 
    - O Value at Risk (VaR) é uma medida de risco que indica a perda máxima esperada, com um determinado nível de confiança, em um intervalo de tempo pré-determinado. \n
    
    ## Parâmetros
    
    - symbol -> Nome do Ativo para fazer a busca \n
    - confidence_level -> Nivel de confiança para o VAR (0 a 1), normalmente usado em 0.95 \n
    - lookback_period -> Periodo EM DIAS a ser considerado para o cálculo do VaR

    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO, SENAO VAI PASSAR
        tipoInfo = type(info)
        if tipoInfo == dict:
            pass
    except:
         print("Invalid ticker symbol")

    # Obter os dados de preços do ativo
    prices = stock.history(period=f"{lookback_period}d")["Close"]

    # Calcular o retorno diário da ação
    returns = np.log(prices / prices.shift(1))

    # Calcular o desvio padrão e o VaR histórico
    std_dev = returns.std()
    var = std_dev * norm.ppf(1 - confidence_level)

    return print({"VaR": round(var * prices[-1], 2)})

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/VaR")
responseHistory = Response(media_type="application/json")


## CARTEIRA DE ATIVOS ##

@app.get("stocks/{symbol}/AnnualReturn", response_model=None)
def asset_portfolio(symbols: Union[str, list], start_date: str, end_date: str) -> pd.DataFrame:
    """
    ## Usabilidade
    - Recebe uma lista e retorna um DataFrame com as informações dos ativos e algumas estatísticas básicas. \n
    
    ## Parâmetros
    - symbols -> Recebe uma lista ou um unico ativo para buscar na base \n
    - start_date -> Data de Inicio da busca das infos (preco, volume, etc) do ativo \n
    - end_date -> Data Final para a busca das infos (preco, volume, etc) do ativo \n
    
    """
    # Importar dados dos ativos
    
    if isinstance(symbols, str):
        print(f"Você digitou uma string: {symbols}")
        dados = yf.download(tickers= symbols, start= start_date, end= end_date, group_by= 'ticker')
            
        try:
            #Exibe o preço a mercado da açao
            data = yf.Ticker(symbols).info
            valueMarket = data['regularMarketPrice']
            
            #Retorna o preço de fechamento da açao
            close = dados['Close']
            
            # Calcular retornos diários
            retorno_diario = close.pct_change()
            
            # Calcular retornos anuais
            retorno_anual = retorno_diario.mean() * 252
                    
            # Calcular desvio padrão anual
            desvio_padrao_anual = retorno_diario.std() * (252 ** 0.5)
            
            # Calcular valor total investido
            valor_investido = close.iloc[0] * 100
            
            # Calcular valor atual
            valor_atual = close.iloc[-1] * 100
            
            # Calcular retorno total
            retorno_total = (valor_atual - valor_investido) / valor_investido
            
            # Organizar em um DataFrame
            valueSymbols = pd.DataFrame({
                'Ativo' : symbols,
                'Preço a Mercado' : valueMarket,
                'Retorno anual': retorno_anual,
                'Desvio padrão anual': desvio_padrao_anual,
                'Retorno total': retorno_total
            }, index=[1])
            
            print(valueSymbols)

        except:
            print("Ticker inválido")
  
    elif isinstance(symbols, list):
        valueDF = pd.DataFrame()
        print(f"Você digitou uma lista: {symbols}")
        for simbolo in symbols:
            dados = yf.download(tickers= simbolo, start= start_date, end= end_date, group_by= 'ticker')
        
            # Selecionar preços de fechamento
            close = dados['Close']
            
            # Calcular retornos diários
            retorno_diario = close.pct_change()
            
            # Calcular retornos anuais
            retorno_anual = retorno_diario.mean() * 252
                        
            # Calcular desvio padrão anual
            desvio_padrao_anual = retorno_diario.std() * (252 ** 0.5)
            
            # Calcular valor total investido
            valor_investido = close.iloc[0] * 100
            
            # Calcular valor atual
            valor_atual = close.iloc[-1] * 100
            
            # Calcular retorno total
            retorno_total = (valor_atual - valor_investido) / valor_investido
            
            # Organizar em um DataFrame
            returnSymbols = pd.DataFrame({
                'Ativo' : simbolo,
                'Retorno anual': retorno_anual,
                'Desvio padrão anual': desvio_padrao_anual,
                'Retorno total': retorno_total
            }, index=[len(symbols)])
            
            valueDF = pd.concat([returnSymbols, valueDF])
        print(valueDF)

    else:
        print("Tipo inválido. Digite uma string ou uma lista.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="stocks/{symbol}/AnnualReturn")
responseHistory = Response(media_type="application/json")


## ALOCAÇAO DE MARKOWITZ ##

@app.get("stocks/{symbol}/MarkowitzAllocationn")
def markowitz_allocation(symbols: list, star_date: str, end_date: str) -> dict: 
    
    """
    ## Usabilidades 
    Alocação de Markowitz é uma técnica de otimização de portfólio que visa encontrar a combinação ideal de ativos para maximizar o retorno do investimento enquanto minimiza o risco. \n

    ## O Retorno Esperado
    - representa a taxa de retorno média que se espera obter do portfólio de investimentos \n
    ## O Risco 
    - representa a medida de volatilidade do portfólio, ou seja, 
    quanto mais instável for o retorno dos ativos, maior será o risco do portfólio como um todo \n
    
    
    
    ## Parâmetros
    
    - symbols -> Recebe uma lista de ativos para buscar na base \n
    - start_date -> Data de Inicio da busca das infos (preco, volume, etc) do ativo \n
    - end_date -> Data Final para a busca das infos (preco, volume, etc) do ativo \n

    """
    dados = yf.download(symbols, start=star_date, end=end_date)['Adj Close']

    # Calculando os retornos diários dos ativos
    retornos = dados.pct_change().dropna()

    # Calculando a matriz de covariância dos retornos
    matriz_covariancia = retornos.cov()

    # Definindo o vetor de pesos de igual peso para todos os ativos
    pesos = np.array([1/len(symbols)] * len(symbols))

    # Calculando o retorno esperado e o risco da carteira com pesos iguais
    retorno_esperado = np.sum(retornos.mean() * pesos) * 252
    
    # adiciona uma pequena constante à diagonal da matriz de covariância, de forma que ela deixe de ser singular. 
    risco = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia, pesos))) * np.sqrt(252)
    
    # Calculando a alocação de Markowitz
    lambda_ = 0.1 * np.trace(matriz_covariancia)  # Constante de regularização
    cov_inv = np.linalg.inv(matriz_covariancia + lambda_ * np.eye(matriz_covariancia.shape[0]))    
    vetor_uns = np.ones((len(symbols),1))
    w_markowitz = np.dot(cov_inv, vetor_uns) / np.dot(np.dot(vetor_uns.T, cov_inv), vetor_uns)
    w_markowitz = w_markowitz.flatten()

    # Imprimindo a alocação de Markowitz
    markowitzList = []
    for i in range(len(symbols)):
        taxas = f"O ativo {symbols[i]} deve ser alocado em {w_markowitz[i] * 100:.2f}% da carteira"
        markowitzList.append(taxas)
    json_data = {'Retorno Esperado' : retorno_esperado,
                 'Risco da Carteira' : risco,
                 'Alocacao Markowitz' : markowitzList}
    
    formatted_json = json.dumps(json_data, indent=4, sort_keys=True)
    print(formatted_json)
    
    return formatted_json
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="stocks/{symbol}/MarkowitzAllocation")
responseHistory = Response(media_type="application/json")


## BUSCA INFO DE FUNDOS ##

@app.get("/infoFunds", response_model=None)
def get_funds(symbol: str) -> pd.DataFrame:
    url = "https://www.fundsexplorer.com.br/ranking"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find_all("table")[0]
    fundsDF = pd.read_html(str(table))[0]    
    fundsDF['Código do fundo'] = fundsDF['Código do fundo'].apply(lambda x: x+'.SA')
        
    valuesFI = fundsDF.loc[(fundsDF['Código do fundo'] == symbol)]
    valuesFI = valuesFI[['Código do fundo', 'Setor', 'Preço Atual', 'Dividendo', 'Variação Preço', "Rentab. Período"]]
    print(valuesFI)
    return valuesFI

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="/infoFunds")
responseHistory = Response(media_type="application/json")


## COMPARADOR DE FUNDOS COM BASE NO SETOR ##

@app.get("/compareSetorFunds", response_model=None)
def compare_setor_funds(setor= TipoSetores, rentabilidade_min = 0) -> pd.DataFrame:
    
    """
    ## Usabilidade
    
    - Funçao que utiliza as metricas e medias dos fundo com base no seu Setor para uma analise mais restrita
    
    ## Parâmetros
    
    - rentabilidade_min -> Valor em % para buscar a rentabilidade minima do fundo escolhido
    - setor -> Setores de fundos que poderam ser escolhidos, segue a lista:
    
    ```
    TiposSetores:
    - Corporativas = "Lajes Corporativas" 
    - Mobiliarios = "Títulos e Val. Mob."
    - Shoppings = "Shoppings"
    - Hibridos = 'Híbrido'
    - Renda = 'Renda'
    - Logistica = 'Logística'
    - Hospital = 'Hospital'
    - Residencial = 'Residencial'
    - Outros = 'Outros'

    ```
    
    ## Exemplo:
    
    ```
    >>> bb.compare_setor_funds(setor='Corporativas', rentabilidade_min = 3)
    ```
    
    
    """
    
    url = "https://www.fundsexplorer.com.br/ranking"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find_all("table")[0]
    fundsDF = pd.read_html(str(table))[0]

    formataValoresNumero(fundsDF, "Rentab. Período")
    rentabilidade_min = rentabilidade_min / 100
    valuesFI = fundsDF.loc[(fundsDF["Rentab. Período"]/100) > rentabilidade_min]
    valuesFI = valuesFI[['Código do fundo', 'Setor', 'Preço Atual', 'Dividendo', 'Variação Preço', "Rentab. Período"]]
    valuesFI = valuesFI.dropna()
    rentabilidade_media = valuesFI['Rentab. Período'].mean()
    rentabilidade_mercado = valuesFI.loc[valuesFI["Setor"] == setor]["Rentab. Período"].mean()
    
    desvio_padrao = valuesFI["Rentab. Período"].std()
    
    resultados = pd.DataFrame({
        "Rentabilidade Média dos FIIs Selecionados": [rentabilidade_media],
        "Rentabilidade Média do Mercado": [rentabilidade_mercado],
        "Desvio Padrão das Rentabilidades dos FIIs Selecionados": [desvio_padrao]
    })
    
    resultados = resultados.fillna('O setor/valor nao foi encontrado')
    print(resultados)
    return resultados

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="/compareSetorFunds")
responseHistory = Response(media_type="application/json")


## COMPARADOR DE FUNDOS ##

@app.get("/compareFunds", response_model=None)
def compare_funds(listfund= None, fund_1= str, fund_2= str) -> pd.DataFrame:
    """
    ## Usabilidade 
    
    - Funçao que realiza a comparaçao entre dois fundos, seja feita a requisiçao dos fundos via lista ou unicos 
    - Requisiçao Listas: Retorna o fundo com maior porcentagem de risco (a variação percentual dos preços dos ativos, calculo realizado com base no desvio padrão)
    - Requisiçao Unica: Retorna um Dataframe com as principais informaçoes dos fundos, afim de uma comparaçao entre seus valores \n
    
    ## Parâmetros
    
    - listfund -> Lista dos fundos para analise de risco
    - fund_1 -> Primeiro fundo para analise unica
    - fund_2 -> Segundo fundo para analise unica
    
    ## Exemplos:
    
    ```
    >>> bb.compare_funds(listfund= list) 
                    ou
    >>> bb.compare_funds(fund_1= 'fund1', fund_2= 'fund2')
    ```
    
    """

    if fund_1 and fund_2 != None:
        url = "https://www.fundsexplorer.com.br/ranking"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find_all("table")[0]
        fundsDF = pd.read_html(str(table))[0]    
        fundsDF['Código do fundo'] = fundsDF['Código do fundo'].apply(lambda x: x+'.SA')
        fundsDF = fundsDF[['Código do fundo', 'Setor', 'Preço Atual', 'Dividendo', 'Variação Preço', "Rentab. Período", 'Dividend Yield', 'DY (3M) Acumulado']]
        fundsDF = fundsDF.drop_duplicates(subset=['Código do fundo'])
        
        fund1 = fundsDF.loc[fundsDF["Código do fundo"] == fund_1]
        fund2 = fundsDF.loc[fundsDF["Código do fundo"] == fund_2]

        unit = pd.concat([fund1, fund2])
        if unit.empty:
            print('Nao foram apresentado dados dos fundos para verificaçao unica')
        else:
            print(unit)
        
    if listfund is None:
        listfund = []
    else:
        max_risco = -1
        ticker_max_risco = ''
        for ticker in listfund:
            fundo = yf.Ticker(ticker)
            df = fundo.history(period='max')
            desvio_padrao = df['Close'].pct_change().std()
            if desvio_padrao > max_risco:
                max_risco = desvio_padrao
                ticker_max_risco = ticker
            
        valuerisk = pd.DataFrame({'Fund' : ticker_max_risco,
                    'Max risk (%)' : max_risco * 100}, index=[len(listfund)])
        
        if valuerisk.empty:
            print('Nao foram apresentado dados dos fundos para verificaçao múltipla')
        else:
            print(valuerisk)
            
    return valuerisk, unit
            
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="/compareFunds")
responseHistory = Response(media_type="application/json")


## SIMULADOR DE AÇOES ##

@app.get("/bestAssets", response_model=None)
def best_assets(perfil= TipoPerfis) -> pd.DataFrame:
    
    """
    ## Usabilidade
    - Função que analisa os principais ativos listados no mercado que com base no perfil escolhido mostra quais podem ser suas escolhas e quantos porcento se deve ter na carteira
    
    ## Parâmetros
    
    - perfil -> Perfis que podem ser escolhidos para realizar a análise, segue a lista: \n
    
    ```
    TipoPerfis:
    * Agressivo
    * Moderado
    * Conservador
    
    ```
    
    ## Exemplo
    
    ```
    
    >>> alocation = best_assets(perfil='Agressivo')
    
    ```
    
    """
    
    # Lista de ativos
    url = "https://www.dadosdemercado.com.br/bolsa/acoes"
    response = requests.get(url)
    if response.status_code != 200:
        print('Acesso negado a base, tente novamente mais tarde.')
    else:
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find_all("table")[0]
        fundsDF = pd.read_html(str(table), decimal=',', thousands='.')[0]    
        fundsDF['Código'] = fundsDF['Código'].apply(lambda x: x+'.SA')
        print('Buscando ativos....')
        precos = yf.download(fundsDF['Código'].tolist(), period="1y")['Close']

        # Função para calcular o retorno diário médio
        def calcular_retorno(precos):
            retorno = precos.pct_change()
            retorno_medio = retorno.mean()
            return retorno_medio

        # Função para calcular a volatilidade diária
        def calcular_risco(precos):
            retorno = precos.pct_change()
            risco = retorno.std()
            return risco

        # Calcular o retorno e o risco de cada ativo
        retorno = {}
        risco = {}
        for ativo in precos.columns:
            precos_ativo = precos[ativo]
            retorno[ativo] = calcular_retorno(precos_ativo)
            risco[ativo] = calcular_risco(precos_ativo)

        # Análise para um cliente agressivo
        if perfil == 'Agressivo':
            agressivo = []
            for ativo in precos.columns:
                if retorno[ativo] > 0.0001 and risco[ativo] > 0.01:
                    agressivo.append(ativo)
            # print('Para um cliente agressivo, Ativos selecionados:', agressivo)
            DfAgressivo = pd.DataFrame(agressivo, columns=['Ativos P/Agressivo'])
            print('Realizando calculos para a sua carteira com base no seu perfil.')
            alocation_Agressive = markowitz_allocation(agressivo, star_date= oneY, end_date= currently )
            dataAlocation_Agressive = json.loads(alocation_Agressive)
            dataAlocation_Agressive = pd.DataFrame(dataAlocation_Agressive)

            # adiciona as quebras de linha na coluna "Alocacao Markowitz"
            dataAlocation_Agressive['Alocacao Markowitz'] = dataAlocation_Agressive['Alocacao Markowitz'].replace('\n', '\\n', regex=True)
            dataAlocation_Agressive['Retorno Esperado'] = dataAlocation_Agressive['Retorno Esperado'].drop_duplicates().dropna()
            dataAlocation_Agressive['Risco da Carteira'] = dataAlocation_Agressive['Risco da Carteira'].drop_duplicates().dropna()
            
            if dataAlocation_Agressive.empty:
                pass
            else:
                return dataAlocation_Agressive
            
        elif perfil == 'Moderado':
            # Análise para um cliente moderado
            moderado = []
            for ativo in precos.columns:
                if retorno[ativo] > 0.0003 and risco[ativo] < 0.03:
                    moderado.append(ativo)
            # print('Para um cliente moderado, Ativos selecionados:', moderado)
            DfModerado = pd.DataFrame(moderado, columns=['Ativos P/Moderado'])
            print('Realizando calculos para a sua carteira com base no seu perfil.')
            alocation_Moderade = markowitz_allocation(moderado, star_date= oneY, end_date= currently )
            dataAlocation_Moderade = json.loads(alocation_Moderade)
            dataAlocation_Moderade = pd.DataFrame(dataAlocation_Moderade)

            # adiciona as quebras de linha na coluna "Alocacao Markowitz"
            dataAlocation_Moderade['Alocacao Markowitz'] = dataAlocation_Moderade['Alocacao Markowitz'].replace('\n', '\\n', regex=True)
            dataAlocation_Moderade['Retorno Esperado'] = dataAlocation_Moderade['Retorno Esperado'].drop_duplicates().dropna()
            dataAlocation_Moderade['Risco da Carteira'] = dataAlocation_Moderade['Risco da Carteira'].drop_duplicates().dropna()
            
            if dataAlocation_Moderade.empty:
                pass
            else:
                return dataAlocation_Moderade
            
        elif perfil == 'Conservador':
            # Análise para um cliente conservador
            conservador = []
            for ativo in precos.columns:
                if risco[ativo] < 0.01:
                    conservador.append(ativo)
            DfConservador = pd.DataFrame(conservador, columns=['Ativos P/Conservador'])
            # print('Para um cliente conservador, Ativos selecionados:', conservador)
            print('Realizando calculos para a sua carteira com base no seu perfil.')
            alocation_Conservative = markowitz_allocation(conservador, star_date= oneY, end_date= currently )
            dataAlocation_Conservative = json.loads(alocation_Conservative)
            dataAlocation_Conservative = pd.DataFrame(dataAlocation_Conservative)

            # adiciona as quebras de linha na coluna "Alocacao Markowitz"
            dataAlocation_Conservative['Alocacao Markowitz'] = dataAlocation_Conservative['Alocacao Markowitz'].replace('\n', '\\n', regex=True)
            dataAlocation_Conservative['Retorno Esperado'] = dataAlocation_Conservative['Retorno Esperado'].drop_duplicates().dropna()
            dataAlocation_Conservative['Risco da Carteira'] = dataAlocation_Conservative['Risco da Carteira'].drop_duplicates().dropna()
            
            if dataAlocation_Conservative.empty:
                pass
            else:
                return dataAlocation_Conservative
        else:
            print('Perfil não reconhecido, os perfis disponiveis estao presentes na explicação da função')
            
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="/bestAssets")
responseHistory = Response(media_type="application/json")


@app.get("/bestAssetsValues", response_model=None)
def best_assets_value(valor= 0) -> pd.DataFrame:
    
    """
    ## Usabilidade
    - Função que analisa os ativos do IBOVESPA que com base no valor de investimento escolhido mostra quais podem ser suas escolhas, o quanto ira ter que investir para cada ativo e o retorno aproximado para cada um deles \n
    - Usamos como metodo de cálculo o a medida Sharpe que nada mais é que uma  medida de desempenho de investimentos que leva em consideração o retorno obtido pelo investimento em relação ao risco assumido
    ## Parâmetros
    
    - valor -> Valor do investimento, por padrão 0

    """
    
    # Lista de ativos
    ativos = pd.read_excel(r'C:\Users\Luis\ProjetoFin\BBFinance\Data\AtivosIbov.xlsx')
    ativos['Código'] = ativos['Código'].apply(lambda x: x+'.SA')

    # Baixa os dados históricos dos ativos nos últimos 12 meses
    precos = yf.download(ativos['Código'].tolist(), period='1y')['Close']

    # Remove colunas com valores ausentes
    precos = precos.dropna(axis=1)

    # Calcula o retorno esperado e o risco dos ativos
    retorno_esperado = precos.pct_change().mean() * 252
    risco = precos.pct_change().std() * np.sqrt(252)

    # Cria um dataframe com os dados dos ativos
    ativos_df = pd.DataFrame({'retorno_esperado': retorno_esperado, 'risco': risco})

    # Adiciona uma coluna com o índice de Sharpe de cada ativo
    ativos_df['sharpe'] = ativos_df['retorno_esperado'] / ativos_df['risco']

    # Ordena os ativos por índice de Sharpe e seleciona os 5 melhores
    ativos_df = ativos_df.sort_values('sharpe', ascending=False).head(6)

    # Calcula a alocação de cada ativo na carteira
    ativos_df['alocacao'] = ativos_df['sharpe'] / ativos_df['sharpe'].sum()

    # Calcula o valor alocado em cada ativo com base no valor sugerido
    ativos_df['valor'] = ativos_df['alocacao'] * valor
    ativos_df = ativos_df.reset_index()
    
    ativos_df = ativos_df.rename(columns={'index' : 'Ativos', 'valor' : 'Qtd Necessaria (R$)'})
    
    for ativo in ativos_df['Ativos']:
        data = yf.download(ativo, period="1y")
        retorno_esperado[ativo] = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]

    # calcular o retorno para cada ativo baseado no valor investido
    retorno_por_ativo = {ativo: retorno_esperado[ativo] * valor for ativo in ativos_df['Ativos'] }
    ListaRetorno = retorno_por_ativo.values()
    ListaRetorno = list(ListaRetorno)
    
    ativos_df['Retorno Aprox.'] = ListaRetorno
    
    # Retorna o dataframe com os ativos selecionados e seus valores alocados
    return ativos_df[['Ativos', 'Qtd Necessaria (R$)', 'Retorno Aprox.']]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="/bestAssetsValues")
responseHistory = Response(media_type="application/json")


#################################################################

def prever_taxa_retorno(ativo):
    
    dados_historicos = yf.download('PETR4.SA', period='1y')
    dados_historicos = dados_historicos.reset_index()

    # Seleciona os dados históricos do ativo
    # dados_ativo = dados_historicos.loc[dados_historicos['Ativo'] == ativo]

    # Separa as variáveis independentes (datas) e dependentes (preços)
    datas = pd.to_numeric(dados_historicos['Date'].astype(str).str.replace('-', ''))
    precos = dados_historicos['Close']

    # Cria o modelo de regressão linear
    modelo = LinearRegression()

    # Treina o modelo com os dados históricos
    modelo.fit(datas.values.reshape(-1, 1), precos)

    # Faz a previsão da taxa de retorno para o próximo período
    data_atual = pd.Timestamp.today().strftime('%Y-%m-%d')
    data_futura = (pd.Timestamp.today() + pd.DateOffset(days=365)).strftime('%Y-%m-%d')
    taxa_retorno = (modelo.predict(pd.to_numeric([data_futura.replace('-', '')]).reshape(-1, 1)) / 
                    modelo.predict(pd.to_numeric([data_atual.replace('-', '')]).reshape(-1, 1)) - 1)

    return taxa_retorno[0] * -1


# QUANTO MAIOR A TAXA DE RETORNO EM MENOS TEMPO VAI TER O RETORNO EM DIVIDENDO, A QUESTAO É PEGAR UMA TAXA DE RETORNO JUSTA OU UMA PADRAO

def retornos_dividendos(ativos: Union[str, list], investimento: Union[int, float], taxa_desconto= 0.23):
    data = pd.DataFrame(columns=['Ativo', 'Retorno com Dividendos', 'Tempo para Atingir Retorno'])
    
    if isinstance(ativos, str):
        ticker = yf.Ticker(ativos)
        preco = ticker.history(period="1y")['Close'][0]
        dividendos = ticker.dividends.sum()
        retorno = ((preco + dividendos) / preco - 1) * investimento
        tempo = np.log(retorno / investimento + 1) / np.log(1 + 0.01 * taxa_desconto) / 12  # tempo em meses
        data = data.append({'Ativo': ativos, 'Retorno com Dividendos': retorno, 'Tempo para Atingir Retorno': tempo}, ignore_index=True)
    else:    
        for ativo in ativos:
            ticker = yf.Ticker(ativo)
            preco = ticker.history(period="1y")['Close'][0]
            dividendos = ticker.dividends.sum()
            retorno = ((preco + dividendos) / preco - 1) * investimento
            tempo = np.log(retorno / investimento + 1) / np.log(1 + 0.01 * taxa_desconto) / 12  # tempo em meses
            data = data.append({'Ativo': ativo, 'Retorno com Dividendos': retorno, 'Tempo para Atingir Retorno': tempo}, ignore_index=True)

    return data

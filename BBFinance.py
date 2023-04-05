import yfinance as yf
from scipy.stats import norm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from fastapi import FastAPI, Response
from fastapi.templating import Jinja2Templates
import json
import uvicorn
import pandas as pd
import numpy as np
yf.pdr_override()
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

## INFOS DAS AÇOES ##

def get_info(symbol: str):
    
    """
    Usabilidade -> Busca as principais informaçoes sobre o ativo selecionado como Preço e Dividendos \n
    
    symbol -> Nome do Ativo para a busca \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO
        if isinstance(info, dict):
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
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}")
response = Response(media_type="application/json")


## HISTORICO DAS AÇOES ##

@app.get("/stocks/{symbol}/history")
def get_stock_history(symbol: str, period: str = '1y'):
    
    """
    Usabilidade -> Usada para verificar o histórico da açao selecionada e em qual periodo \n
    
    symbol -> Nome do Ativo para a busca \n
    period -> Data em ANOS para a busca das informaçoes do Ativo \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO
        if isinstance(info, dict):
            pass #SE FOR VAI SO PASSAR 
    except:
        return {"error": print("Invalid ticker symbol")}
    
    history = stock.history(period=period)
    
    if history.empty:
        return {"error": print("No data found")}
    else:
        history_dict = history.to_dict(orient="list")
        history_df = pd.DataFrame.from_dict(history_dict).reset_index(drop=False)     
        print(history_df)
        # json_data = {'symbol': symbol,
        # "history":  history_df.to_dict(orient="records"),
        # }

        # formatted_json = json.dumps(json_data, indent=2)
        # print(formatted_json)

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/history")
responseHistory = Response(media_type="application/json")


## TENDENCIA DE PREÇO ##

@app.get("/stock/{symbol}/trend")
def get_stock_trend(symbol: str):
    
    """
    Usabilidade -> Identifica a tendencia de preço de uma açao, se ira ser de ALTA ou BAIXA
    
    symbol -> Nome do Ativo para a busca \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO
        if isinstance(info, dict):
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

@app.get("/stock/{symbol}/technical")
def get_stock_technicals(symbol: str):
    
    """
    Usabilidade -> cálculo envolve a comparação da média de ganhos em um período de tempo com a média de perdas em um período de tempo. \n
    Como interpretar -> Quando o RSI está acima de 70, o ativo é considerado sobrecomprado, o que significa que pode estar prestes a sofrer uma correção para baixo. 
    Quando o RSI está abaixo de 30, o ativo é considerado sobrevendido, o que significa que pode estar prestes a subir novamente. \n 
    
    symbol -> Nome do Ativo para a busca \n
    
    """
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO, SENAO VAI PASSAR
        if isinstance(info, dict):
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

@app.get("stocks/{symbol}/volatility")
def get_volatility(symbol: str, start_date: str, end_date: str):
    
    """
    Usabilidade -> Método usado para verificar a volatilidade de um ativo em comparacao ao mercado em que esta  \n
    
    symbol -> Nome do Ativo para a busca \n
    start_date -> Data de Inicio da busca das infos (preco, volume, etc) do ativo \n
    end_date -> Data Final para a busca das infos (preco, volume, etc) do ativo \n
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

@app.get("stocks/{symbol}/beta")
def get_beta(symbol: str):
    
    """
    Usabilidade -> O beta é uma medida estatística que indica a relação entre a volatilidade de uma ação e a volatilidade do mercado como um todo.
    O valor do beta é utilizado para medir o risco de uma ação em relação ao mercado em que ela é negociada. \n
    
    symbol -> Nome do Ativo para a busca \n
    market -> Como padrao, Mercado: IBOVESPA / BVSP
    """
    
    # Obter os dados do ativo e do mercado
    try:
        asset = yf.Ticker(symbol)
        market = yf.Ticker("^BVSP") # Índice Bovespa como mercado de referência
        info = asset.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO, SENAO VAI PASSAR
        infoMarket = market.info
        if isinstance(info, dict) | isinstance(infoMarket, dict) :
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

def calculate_historical_var(symbol: str, confidence_level: float, lookback_period: int) -> float:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info #DADO Q VEM COMO UM DICIONARIO, SE NAO FOR UM DICIONARIO VAI APRESENTAR TICKER INVALIDO, SENAO VAI PASSAR
        if isinstance(info, dict):
            pass 
    except:
        return {"error": print("Invalid ticker symbol")}

    # Obter os dados de preços do ativo
    prices = stock.history(period=f"{lookback_period}d")["Close"]

    # Calcular o retorno diário da ação
    returns = np.log(prices / prices.shift(1))

    # Calcular o desvio padrão e o VaR histórico
    std_dev = returns.std()
    var = std_dev * norm.ppf(1 - confidence_level)

    return round(var * prices[-1], 2)

@app.get("stocks/{symbol}/VaR")
def var(symbol: str, confidence_level: float, lookback_period: int):
    
    """
    Usabilidade -> O Value at Risk (VaR) é uma medida de risco que indica a perda máxima esperada, com um determinado nível de confiança, em um intervalo de tempo pré-determinado. \n
    
    symbol -> Nome do Ativo para fazer a busca \n
    confidence_level -> Nivel de confiança para o VAR (0 a 1), normalmente usado em 0.95 \n
    lookback_period -> Periodo EM DIAS a ser considerado para o cálculo do VaR

    """
    
    return {"VaR": print(calculate_historical_var(symbol, confidence_level, lookback_period))}

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000, default="stocks/{symbol}/VaR")
responseHistory = Response(media_type="application/json")


@app.post("/quote")
def get_quote(tickers: list):
    """
    Recebe uma lista de tickers via método POST e retorna a cotação atual de cada um via método GET.
    """
    response = {}
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).info
            response[ticker] = data['regularMarketPrice']
        except:
            response[ticker] = "Ticker inválido"

    json_data = {'symbol': response}
    
    formatted_json = json.dumps(json_data, indent=2)
    print(formatted_json)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, default="/quote")
responseHistory = Response(media_type="application/json")

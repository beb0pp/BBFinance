# BBFinance

Biblioteca de análise financeira

## Descrição

Esta biblioteca fornece funções e classes úteis para a análise de dados financeiros, incluindo cotações de ações, índices de mercado e outras informações relevantes para o mercado financeiro.

## GitHub

Segue o link para o repositorio do GitHub -> https://github.com/beb0pp/BBFinance

## Funcionalidades

* Recuperação de cotações de ações de diversas fontes
* Cálculo de indicadores financeiros, como Tendência de Preços, RSI, VaR e Alocação de Markowitz
* Visualização de gráficos de séries temporais e análise técnica
* Análise de risco e retorno de carteiras de investimento

## Instalação

Para instalar a biblioteca, use o seguinte comando:

`pip install BBFinance`

## Comandos

* `bb.get_info()` - Busca as principais informações dos ativos
* `get_stock_history() `- Busca o histórico de preços do ativo inserido
* `get_stock_trend()` - Identifica a tendencia de preço de uma açao, se ira ser de ALTA ou BAIXA
* `get_stock_technicals()` - Cálculo envolve a comparação da média de ganhos e perdas de um periodo.
* `get_volatility()` - Método usado para verificar a volatilidade de um ativo em comparacao ao mercado.
* `get_beta()` - O valor do beta é utilizado para medir o risco de uma ação em relação ao mercado em que ela é negociada
* `get_var()` - O Value at Risk (VaR) é uma medida de risco que indica a perda máxima esperada, com um determinado nível de confiança, em um intervalo de tempo pré-determinado.
* `asset_portfolio()` - Recebe uma lista e retorna um DataFrame com as informações dos ativos e algumas estatísticas básicas.
* `markowitz_allocation()` - Alocação de Markowitz é uma técnica de otimização de portfólio que visa encontrar a combinação ideal de ativos para maximizar o retorno do investimento enquanto minimiza o risco.
* `get_funds()` - Função utilizada para adquirir as principais caracteristicas e informações do fundo selecionado
* `compare_setor_funds()` - Função que utiliza as métricas e médias dos fundo com base no seu Setor para uma análise mais restrita
* `compare_funds()` - Função que realiza a comparaçao entre dois fundos, seja feita a requisiçao dos fundos via lista ou únicos
* `best_assets()` - Função que analisa os principais ativos listados no mercado que com base no perfil escolhido mostra quais podem ser suas escolhas e quantos porcento se deve ter na carteira
* `best_assets_value()` - Função que análisa os ativos do IBOVESPA que com base no valor de investimento escolhido mostra quais podem ser suas escolhas, o quanto ira ter que investir para cada ativo e o retorno aproximado para cada um deles

## Exemplos

```
import BBFinance as bb

>>> info = bb.get_info('PETR4.SA')

>>> info = bb.get_stock_history(symbol= 'PETR4.SA', period= '1y')

>>> info = bb.get_stock_trend(symbol= 'PETR4.SA')

>>> info = bb.get_stock_technicals(symbol= 'PETR4.SA')

>>> info = bb.get_volatility(symbol= 'PETR4.SA', start_date= '2023-01-01', end_date= '2023-12-01')

>>> info = bb.get_beta(symbol= 'PETR4.SA')

>>> info = bb.get_var(symbol= 'PETR4.SA',confidence_level= 0.95, lookback_period= 30)

>>> info = bb.asset_portfolio(symbols= 'PETR4.SA' | ['PETR4.SA', 'VALE3.SA'], start_date= '2023-01-01', end_date= '2023-12-01') #Incluir uma lista de ativos (['PETR4.SA', 'VALE3.SA']) ou ativo unico ('PETR4.SA')

>>> info = bb.markowitz_allocation(symbols= ['PETR4.SA', 'VALE3.SA'], start_date= '2023-01-01', end_date= '2023-12-01')

>>> info = bb.get_funds(symbol= 'PETR4.SA')

>>> info = bb.compare_setor_funds(setor: 'Corporativo', rentabilidade_min = 1)

>>> info = bb.compare_funds(listfund= ['MXRF11.SA', MGFF11.SA] | fund_1= 'MXRF11.SA', fund_2= 'MGFF11.SA') #Incluir uma lista (parametro: listfund) ou fundos separados (parametros: fund_1, fund_2)

>>> info = bb.best_assets(perfil= 'Agressivo')

>>> info = bb.best_assets_value(valor= 1000 | 1320.50) #Incluir um valor inteiro (R$ 1000) ou um valor quebrado (R$ 1320.50)

```

## Contribuições

Contribuições são bem-vindas! Se você deseja contribuir para a biblioteca, por favor, abra uma issue no repositório para discutir a sua ideia ou submeta um pull request.

## Licença

Esta biblioteca é distribuída sob a licença GPLv3. Veja o arquivo LICENSE para mais informações.

## Contato

Para entrar em contato com os mantenedores da biblioteca, envie um e-mail para luss.fel@gmail.com.

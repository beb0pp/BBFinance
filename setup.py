from setuptools import setup

setup(
    name='BBFinance',
    version='1.1.1',
    description='Uma biblioteca com o objetivo de adquirir informaçõe de ações do mercado financeiro de maneira rapida e pratica, afim de incluir todos no mercado',
    url='https://github.com/beb0pp/BBFinance',
    author='Luis Abreu',
    author_email='luss.fel@gmail.com',
    license='MIT',
    packages=['BBFinance'],
    install_requires=[
        'yfinance>=0.2.12',
        'scipy>=1.9.3',
        'selenium>=3.141.0',
        'pandas-datareader>=0.10.0',
        'json5>=0.9.6',
        'numpy>=1.23.4',
        'uvicorn>=0.21.1',
        'fastapi>=0.95.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)

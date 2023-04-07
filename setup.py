from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BBFinance',
    version='1.1.9',
    description='Uma biblioteca com o objetivo de adquirir informações de ações do mercado financeiro de maneira rapida e prática, afim de incluir todos no mercado',
    url='https://github.com/beb0pp/BBFinance',
    author='Luis Abreu',
    author_email='luss.fel@gmail.com',
    license='MIT',
    packages=['BBFinance'],
    long_description= long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'yfinance>=0.2.12',
        'scipy>=1.9.3',
        'selenium>=3.141.0',
        'pandas-datareader>=0.10.0',
        'json5>=0.9.6',
        'numpy>=1.23.4',
        'uvicorn>=0.21.1',
        'fastapi>=0.95.0',
        'scipy>=1.10.1',
        'requests>=2.28.2',
        'bs4>=0.0.1',
        'beautifulsoup4>=4.11.2',
        'pydantic>=1.10.7',
        'Unicode>=1.3.6',
        'typing_extensions>=4.5.0'
        
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)

# Generating Synthetic Financial Time-Series Data Through Generative Adversarial Network

## Abstract

Generative Adversarial Networks (GANs) have emerged as a powerful tool for synthesizing realistic financial time series data by addressing challenges such as data scarcity, privacy, and cost. This paper explores the integration of attention mechanisms, convolutional neural networks (CNNs), and long short-term memory (LSTM) layers within a GAN framework to generate synthetic financial price data. We evaluate the performance of these architectures using three financial instruments: JPMorgan Chase (JPM), Apple Inc. (AAPL), and the MSCI ACWI Index (ACWI). The data spans over two decades of market data. Our results demonstrate that CNN enhanced GANs significantly improve the modeling of temporal dependencies and volatility clustering compared to Attention and LSTM based variants. The synthetic data exhibits statistical properties that align with real world financial time series which is validated by tests for distributional similarity and moments. These findings provide actionable insights for financial institutions seeking to augment datasets for backtesting, risk management, and machine learning applications.

## Report

[PDF Available](https://github.com/mariodematteis/ml-ii-final-project/blob/develop/report/report.pdf)

## Development

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
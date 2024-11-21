## Análise de Sentimentos em Avaliações de Produtos

Este projeto realiza a análise de sentimentos em avaliações de produtos a partir de um dataset de resenhas de clientes. Utilizando técnicas de Processamento de Linguagem Natural (NLP), ele classifica as avaliações em categorias de sentimento, ajudando a entender melhor a percepção dos consumidores em relação aos produtos.

O modelo desenvolvido é baseado em aprendizado de máquina, especificamente utilizando o Naive Bayes ou Support Vector Machine (SVM), e pode ser facilmente integrado em sistemas de análise automatizada de feedback de clientes.

## Funcionalidades

1. Carregamento e Exploração de Dados: O script carrega o dataset de resenhas, realiza uma análise exploratória inicial (verificação de valores nulos, estatísticas descritivas, etc.).
2. Pré-processamento de Texto: Inclui limpeza de texto (remoção de caracteres especiais, conversão para minúsculas), remoção de stopwords, e correção ortográfica.
3. Engenharia de Atributos: Criação de novas features, como a contagem de palavras nas resenhas, que pode melhorar o desempenho do modelo.
4. Modelagem de Sentimentos: O modelo é treinado utilizando o algoritmo Naive Bayes ou Support Vector Machine (SVM) para classificar as resenhas em diferentes categorias de sentimento, como muito positiva, positiva, neutra, negativa e muito negativa.
5. Avaliação do Modelo: O desempenho do modelo é avaliado utilizando métricas como precisão, recall, e F1-Score, para garantir a eficácia na classificação dos sentimentos.
6. Correção ortográfica: O código corrige automaticamente erros de digitação nas avaliações utilizando a biblioteca SpellChecker.
7. Modelo de Machine Learning: O modelo é treinado usando um classificador SVM (Support Vector Machine) e é avaliado com métricas como precisão, recall e f1-score.
8. Salvamento de Modelos: O modelo treinado e o vetor de características são salvos como arquivos .pkl, permitindo o uso posterior ou a implementação de uma API.

## Requisitos

Antes de rodar o projeto, você precisa instalar as dependências. O arquivo requirements.txt lista todas as bibliotecas necessárias.

## Dependências

- pandas: Para manipulação e análise de dados.
- nltk: Para técnicas de processamento de linguagem natural, como remoção de stopwords e lematização.
- spellchecker: Para correção ortográfica do texto.
- scikit-learn: Para as técnicas de aprendizado de máquina, como vetorização de texto e treinamento do modelo.
- joblib: Para salvar e carregar modelos treinados.
- matplotlib: Para visualização de dados e resultados.

## Para instalar as dependências, execute:
 - pip install -r requirements.txt

## Como Rodar o Projeto

## Passo 1: Carregar e Preparar os Dados

O primeiro passo é preparar um arquivo CSV com as colunas review_content (conteúdo da avaliação) e rating (nota atribuída pelo cliente). O arquivo deve ser chamado reviews.csv.

## Passo 2: Treinar o Modelo

Execute o script main.py para treinar o modelo SVM e salvar o modelo treinado e o vetor de características.
 - python main.py

Este processo envolve o pré-processamento das avaliações, a criação de características e o treinamento do modelo. Após a execução, você obterá dois arquivos salvos:

 - sentiment_model_svm.pkl: O modelo SVM treinado.
 - vectorizer.pkl: O vetor de características usado para transformar o texto das resenhas em vetores numéricos.

## Passo 3: Uso do Modelo para Predição

Após treinar o modelo, você pode carregá-lo e realizar previsões em novas resenhas. O código abaixo demonstra como carregar o modelo e vetor de características para fazer previsões:

import joblib

# Carregar o modelo e o vetor de características
model = joblib.load('sentiment_model_svm.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Prever o sentimento de uma nova resenha
review = "This product is amazing, I love it!"
X_new = vectorizer.transform([review])
prediction = model.predict(X_new)

print("Sentimento da resenha:", prediction[0])
# O modelo retornará a categoria de sentimento (por exemplo, positiva, negativa, etc.) para a resenha fornecida.

## Melhorias Futuras

- Modelos mais sofisticados: A integração de modelos mais avançados, como BERT ou GPT, poderia melhorar a precisão da análise de sentimentos, principalmente para capturar contextos mais complexos.
- Análise contextual de sentimentos: Realizar uma análise mais profunda considerando aspectos positivos e negativos dentro de uma mesma avaliação.
- Deploy do modelo: Criar uma API RESTful para disponibilizar o modelo em produção, permitindo que ele seja acessado por outras aplicações.
- Análise de emoções: Além de sentimentos, incluir a análise de emoções (como raiva, felicidade, tristeza) nas avaliações.

## Licença

Este projeto está licenciado sob a MIT License. Veja o arquivo LICENSE para mais detalhes.
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar o lemmatizer e o corretor ortográfico
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Função para corrigir ortografia
def correct_spelling(text):
    return ' '.join([spell.correction(word) if spell.correction(word) else word for word in text.split()])

# Função para lematizar o texto
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Função para limpar o texto
def clean_text(text):
    if not text:  # Verifica se o texto é None ou vazio
        return ""  # Retorna uma string vazia se o texto for vazio ou None
    
    text = text.lower()  # Converter para letras minúsculas
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remover caracteres especiais
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remover stopwords
    text = lemmatize_text(text)  # Lemmatização
    # text = correct_spelling(text)  # Correção ortográfica (removido para melhorar o desempenho)
    return text

# Carregar o dataset
print("Carregando os dados...")
data = pd.read_csv('reviews.csv')

# Visualizar informações gerais do dataset
print("Valores nulos:")
print(data.isnull().sum())
print("\nEstatísticas do dataset:")
print(data.describe(include='all'))
print("\nExemplo de registros:")
print(data.head())
print(data.info())

# Selecionar colunas relevantes
print("\nSelecionando colunas relevantes...")
data = data[['review_content', 'rating']]

# Converter a coluna de rating para numérico
print("Convertendo ratings para numérico...")
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Criar categorias de sentimento
print("Classificando os sentimentos...")
def categorize_sentiment(rating):
    if rating == 5:
        return 'very_positive'
    elif rating == 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    elif rating == 2:
        return 'negative'
    else:
        return 'very_negative'

data['sentiment'] = data['rating'].apply(categorize_sentiment)

# Definir stopwords
stop_words = set(stopwords.words('english'))

# Limpando o texto com paralelização
print("Limpando o texto...")
data['cleaned_review'] = Parallel(n_jobs=-1)(delayed(clean_text)(text) for text in data['review_content'])

# Visualizar os dados limpos
print("\nDados após limpeza:")
print(data[['review_content', 'cleaned_review', 'sentiment']].head())

# Vetorização de texto com TF-IDF
print("\nConvertendo texto para vetores...")
vectorizer = TfidfVectorizer(max_features=1000)  # Limite para 1000 palavras mais frequentes
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

# Dividir dados em treino e teste
print("Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo de Support Vector Machine (SVM)
print("Treinando modelo SVM...")
model_svm = LinearSVC()
model_svm.fit(X_train, y_train)

# Avaliação do modelo
print("Avaliando o modelo...")
y_pred = model_svm.predict(X_test)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# Visualizar distribuição dos sentimentos
print("\nVisualizando a distribuição de sentimentos no dataset...")
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red', 'blue', 'purple'])
plt.title('Distribuição de Sentimentos no Dataset', fontsize=14)
plt.xlabel('Sentimentos', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Salvando o modelo treinado e o vetorizador
print("\nSalvando o modelo e o vetorizador...")
joblib.dump(model_svm, 'sentiment_model_svm.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Modelo e vetorizador salvos com sucesso.")
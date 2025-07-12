"""
Módulo para pré-processamento de texto
"""
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Classe para pré-processamento de texto em português"""
    
    def __init__(self, spacy_model: str = "pt_core_news_sm"):
        """
        Inicializa o preprocessador
        
        Args:
            spacy_model: Nome do modelo spaCy a ser usado
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Modelo spaCy carregado: {spacy_model}")
        except OSError:
            logger.error(f"Modelo spaCy não encontrado: {spacy_model}")
            raise
            
        # Carregar stopwords em português
        try:
            self.stop_words = set(stopwords.words('portuguese'))
        except LookupError:
            logger.warning("Stopwords em português não encontradas. Baixando...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('portuguese'))
        
        # Adicionar stopwords customizadas
        custom_stops = {
            'documento', 'pdf', 'página', 'pagina', 'arquivo', 'texto',
            'sr', 'sra', 'cpf', 'rg', 'cnpj', 'numero', 'número'
        }
        self.stop_words.update(custom_stops)
    
    def clean_text(self, text: str, 
                   remove_numbers: bool = True,
                   remove_special_chars: bool = True,
                   min_length: int = 2) -> str:
        """
        Limpa o texto removendo caracteres indesejados
        
        Args:
            text: Texto a ser limpo
            remove_numbers: Se deve remover números
            remove_special_chars: Se deve remover caracteres especiais
            min_length: Tamanho mínimo das palavras
            
        Returns:
            Texto limpo
        """
        if not text:
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover quebras de linha e espaços extras
        text = re.sub(r'\s+', ' ', text)
        
        # Remover números se solicitado
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remover caracteres especiais se solicitado
        if remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remover palavras muito curtas
        words = text.split()
        words = [word for word in words if len(word) >= min_length]
        
        return ' '.join(words)
    
    def extract_features(self, text: str, 
                        lemmatize: bool = True,
                        remove_stopwords: bool = True) -> List[str]:
        """
        Extrai features do texto usando spaCy
        
        Args:
            text: Texto a ser processado
            lemmatize: Se deve aplicar lemmatização
            remove_stopwords: Se deve remover stopwords
            
        Returns:
            Lista de tokens processados
        """
        if not text:
            return []
        
        # Processar com spaCy
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Pular tokens que são apenas espaços, pontuação ou números
            if (token.is_space or token.is_punct or 
                (token.is_digit and len(token.text) < 4)):
                continue
            
            # Pular stopwords se solicitado
            if remove_stopwords and token.lower_ in self.stop_words:
                continue
            
            # Usar lemma se solicitado, senão usar texto original
            if lemmatize and token.lemma_:
                word = token.lemma_.lower()
            else:
                word = token.lower_
            
            # Filtrar palavras muito curtas
            if len(word) >= 2:
                tokens.append(word)
        
        return tokens
    
    def preprocess(self, text: str, config: dict = None) -> str:
        """
        Aplica pré-processamento completo ao texto
        
        Args:
            text: Texto a ser processado
            config: Configurações de processamento
            
        Returns:
            Texto processado
        """
        if config is None:
            config = {
                'remove_numbers': True,
                'remove_special_chars': True,
                'lemmatize': True,
                'remove_stopwords': True,
                'min_length': 2
            }
        
        # Limpar texto
        cleaned_text = self.clean_text(
            text,
            remove_numbers=config.get('remove_numbers', True),
            remove_special_chars=config.get('remove_special_chars', True),
            min_length=config.get('min_length', 2)
        )
        
        # Extrair features
        tokens = self.extract_features(
            cleaned_text,
            lemmatize=config.get('lemmatize', True),
            remove_stopwords=config.get('remove_stopwords', True)
        )
        
        return ' '.join(tokens)

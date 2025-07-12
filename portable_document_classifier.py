"""
Classificador de Documentos Portável
=====================================

Use este módulo para carregar e usar um modelo pré-treinado
em qualquer projeto Python.

Exemplo de uso:
    from portable_document_classifier import DocumentClassifierPortable
    
    classifier = DocumentClassifierPortable("caminho/para/modelos/")
    resultado = classifier.classify_pdf("documento.pdf")
    print(f"Classe: {resultado['predicted_class']}")
"""

import os
import sys
import logging
import joblib
import fitz  # PyMuPDF
import spacy
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

class DocumentClassifierPortable:
    """
    Classificador de documentos portável que pode ser usado
    em qualquer projeto Python.
    """
    
    def __init__(self, models_dir: str, spacy_model: str = "pt_core_news_sm"):
        """
        Inicializa o classificador portável
        
        Args:
            models_dir: Diretório contendo os arquivos do modelo
            spacy_model: Modelo spaCy a ser usado
        """
        self.models_dir = Path(models_dir)
        self.logger = self._setup_logging()
        
        # Carregar modelo spaCy
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.logger.error(f"Modelo spaCy não encontrado: {spacy_model}")
            self.logger.info("Instale com: python -m spacy download pt_core_news_sm")
            raise
        
        # Carregar modelos treinados
        self._load_models()
        
        self.logger.info("Classificador portável inicializado com sucesso!")
    
    def _setup_logging(self) -> logging.Logger:
        """Configura logging básico"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_models(self):
        """Carrega os modelos treinados"""
        try:
            # Verificar se arquivos existem
            required_files = ['classifier.pkl', 'vectorizer.pkl', 'label_encoder.pkl']
            
            for file in required_files:
                if not (self.models_dir / file).exists():
                    raise FileNotFoundError(f"Arquivo não encontrado: {self.models_dir / file}")
            
            # Carregar modelos
            self.model = joblib.load(self.models_dir / "classifier.pkl")
            self.vectorizer = joblib.load(self.models_dir / "vectorizer.pkl") 
            self.label_encoder = joblib.load(self.models_dir / "label_encoder.pkl")
            
            self.logger.info(f"Modelos carregados de: {self.models_dir}")
            self.logger.info(f"Classes disponíveis: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extrai texto de um arquivo PDF
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído ou None se houver erro
        """
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                self.logger.error(f"Arquivo não encontrado: {pdf_path}")
                return None
            
            # Abrir e extrair texto
            doc = fitz.open(str(pdf_path))
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            
            if not text.strip():
                self.logger.warning(f"Nenhum texto encontrado em: {pdf_path}")
                return None
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair texto de {pdf_path}: {e}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa texto usando as mesmas regras do treinamento
        
        Args:
            text: Texto a ser processado
            
        Returns:
            Texto processado
        """
        if not text:
            return ""
        
        # Limpar texto básico
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Espaços múltiplos
        text = re.sub(r'\d+', '', text)   # Remover números
        text = re.sub(r'[^\w\s]', '', text)  # Remover pontuação
        
        # Processar com spaCy
        doc = self.nlp(text)
        
        # Extrair tokens processados
        tokens = []
        for token in doc:
            if (not token.is_space and 
                not token.is_punct and 
                not token.is_stop and 
                len(token.lemma_) >= 2):
                tokens.append(token.lemma_.lower())
        
        return ' '.join(tokens)
    
    def classify_text(self, text: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Classifica um texto já extraído
        
        Args:
            text: Texto a ser classificado
            min_confidence: Confiança mínima para aceitar classificação
            
        Returns:
            Dicionário com resultado da classificação
        """
        try:
            if not text or len(text) < 50:
                return {
                    'success': False,
                    'error': 'Texto insuficiente para classificação',
                    'text_length': len(text) if text else 0
                }
            
            # Pré-processar texto
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'success': False,
                    'error': 'Texto não pôde ser processado adequadamente'
                }
            
            # Vetorizar
            X = self.vectorizer.transform([processed_text])
            
            # Fazer predição
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Converter para labels
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            max_confidence = float(np.max(probabilities))
            
            # Probabilidades por classe
            class_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probabilities[class_name] = float(probabilities[i])
            
            # Verificar confiança
            classification_accepted = max_confidence >= min_confidence
            
            # Lógica especial para "outros"
            is_others = predicted_label == "outros"
            others_confidence = class_probabilities.get("outros", 0.0)
            
            result = {
                'success': True,
                'predicted_class': predicted_label,
                'confidence': max_confidence,
                'classification_accepted': classification_accepted,
                'min_confidence_threshold': min_confidence,
                'class_probabilities': class_probabilities,
                'text_length_original': len(text),
                'text_length_processed': len(processed_text)
            }
            
            # Tratar categoria "outros"
            if is_others and others_confidence >= 0.8:
                result['predicted_class'] = "documento_nao_reconhecido"
                result['classification_accepted'] = False
                result['rejection_reason'] = f'Documento não reconhecido (confiança "outros": {others_confidence:.3f})'
            elif not classification_accepted:
                result['warning'] = f'Confiança baixa ({max_confidence:.3f} < {min_confidence})'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na classificação: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def classify_pdf(self, pdf_path: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Classifica um arquivo PDF completo
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            min_confidence: Confiança mínima para aceitar classificação
            
        Returns:
            Dicionário com resultado da classificação
        """
        # Extrair texto
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            return {
                'success': False,
                'error': 'Não foi possível extrair texto do PDF',
                'file_path': pdf_path
            }
        
        # Classificar texto
        result = self.classify_text(text, min_confidence)
        result['file_path'] = pdf_path
        
        return result
    
    def batch_classify(self, pdf_directory: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Classifica múltiplos PDFs de um diretório
        
        Args:
            pdf_directory: Diretório com arquivos PDF
            min_confidence: Confiança mínima para aceitar classificação
            
        Returns:
            Lista com resultados das classificações
        """
        directory = Path(pdf_directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {pdf_directory}")
        
        # Encontrar PDFs
        pdf_files = list(directory.glob("*.pdf"))
        self.logger.info(f"Encontrados {len(pdf_files)} arquivos PDF em {directory}")
        
        results = []
        for pdf_file in pdf_files:
            result = self.classify_pdf(str(pdf_file), min_confidence)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            'model_type': type(self.model).__name__,
            'vectorizer_type': type(self.vectorizer).__name__,
            'classes': list(self.label_encoder.classes_),
            'num_features': getattr(self.vectorizer, 'max_features', 'N/A'),
            'models_directory': str(self.models_dir)
        }


# ============================================
# EXEMPLO DE USO
# ============================================

def exemplo_uso():
    """Exemplo de como usar o classificador portável"""
    
    # 1. Inicializar classificador
    classifier = DocumentClassifierPortable("models/")
    
    # 2. Mostrar informações do modelo
    info = classifier.get_model_info()
    print("Informações do Modelo:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # 3. Classificar um PDF
    resultado = classifier.classify_pdf("documento.pdf")
    
    if resultado['success']:
        print(f"📄 Arquivo: {resultado['file_path']}")
        print(f"🎯 Classe: {resultado['predicted_class']}")
        print(f"📊 Confiança: {resultado['confidence']:.3f}")
        print(f"✅ Aceito: {resultado['classification_accepted']}")
        
        print("\nProbabilidades por classe:")
        for classe, prob in resultado['class_probabilities'].items():
            print(f"  {classe}: {prob:.3f}")
    else:
        print(f"❌ Erro: {resultado['error']}")
    
    # 4. Classificar diretório inteiro
    resultados = classifier.batch_classify("documentos/")
    
    print(f"\nResultados em lote: {len(resultados)} documentos")
    for r in resultados:
        if r['success']:
            status = "✅" if r['classification_accepted'] else "⚠️"
            print(f"{status} {r['file_path']}: {r['predicted_class']} ({r['confidence']:.3f})")


if __name__ == "__main__":
    exemplo_uso()
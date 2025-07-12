"""
Módulo de Treinamento do Sistema de Classificação de Documentos
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from tqdm import tqdm
import click

from common.pdf_parser import PDFParser
from common.text_preprocessor import TextPreprocessor
from common.utils import load_config, setup_logging, save_model, get_files_from_directory, create_training_report

class DocumentTrainer:
    """Classe para treinamento do classificador de documentos"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.pdf_parser = PDFParser()
        self.text_preprocessor = TextPreprocessor(
            self.config['models']['spacy_model']
        )
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Configurar logging
        setup_logging(
            self.config['logging']['level'],
            os.path.join(self.config['paths']['logs'], 'training.log')
        )
        self.logger = logging.getLogger(__name__)
    
    def load_training_data(self) -> tuple:
        """
        Carrega dados de treinamento dos diretórios
        
        Returns:
            Tupla com textos e labels
        """
        texts = []
        labels = []
        
        train_dir = Path(self.config['paths']['data_train'])
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Diretório de treinamento não encontrado: {train_dir}")
        
        self.logger.info("Carregando dados de treinamento...")
        
        # Para cada classe de documento
        for doc_class in self.config['document_classes']:
            class_dir = train_dir / doc_class
            
            if not class_dir.exists():
                self.logger.warning(f"Diretório da classe não encontrado: {class_dir}")
                continue
            
            # Obter arquivos PDF da classe
            pdf_files = get_files_from_directory(str(class_dir), ".pdf")
            
            self.logger.info(f"Processando {len(pdf_files)} arquivos da classe '{doc_class}'")
            
            # Processar cada arquivo
            for pdf_file in tqdm(pdf_files, desc=f"Processando {doc_class}"):
                # Extrair texto
                text = self.pdf_parser.extract_text(str(pdf_file))
                
                if text and len(text) >= self.config['text_processing']['min_text_length']:
                    # Pré-processar texto
                    processed_text = self.text_preprocessor.preprocess(
                        text, self.config['text_processing']
                    )
                    
                    if processed_text:
                        texts.append(processed_text)
                        labels.append(doc_class)
                else:
                    self.logger.warning(f"Texto insuficiente em: {pdf_file}")
        
        self.logger.info(f"Carregados {len(texts)} documentos para treinamento")
        
        if len(texts) == 0:
            raise ValueError("Nenhum documento válido encontrado para treinamento")
        
        return texts, labels
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """Cria e configura o vetorizador TF-IDF"""
        return TfidfVectorizer(
            max_features=self.config['models']['vectorizer_max_features'],
            min_df=self.config['models']['min_df'],
            max_df=self.config['models']['max_df'],
            ngram_range=(1, 2),  # Usar unigramas e bigramas
            strip_accents='unicode',
            lowercase=True,
            stop_words=None  # Já removemos stopwords no preprocessamento
        )
    
    def create_classifier(self, classifier_type: str):
        """
        Cria classificador baseado no tipo especificado
        
        Args:
            classifier_type: Tipo do classificador
            
        Returns:
            Instância do classificador
        """
        if classifier_type == "naive_bayes":
            return MultinomialNB(alpha=1.0)
        elif classifier_type == "svm":
            return SVC(kernel='linear', probability=True, random_state=42)
        elif classifier_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Tipo de classificador não suportado: {classifier_type}")
    
    def train(self) -> dict:
        """
        Executa o treinamento completo
        
        Returns:
            Dicionário com resultados do treinamento
        """
        self.logger.info("Iniciando treinamento...")
        
        # Carregar dados
        texts, labels = self.load_training_data()
        
        # Codificar labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Criar e treinar vetorizador
        self.logger.info("Criando representação TF-IDF...")
        self.vectorizer = self.create_vectorizer()
        X = self.vectorizer.fit_transform(texts)
        
        # Criar e treinar classificador
        classifier_type = self.config['models']['classifier_type']
        self.logger.info(f"Treinando classificador: {classifier_type}")
        
        self.model = self.create_classifier(classifier_type)
        self.model.fit(X, encoded_labels)
        
        # Validação cruzada
        self.logger.info("Executando validação cruzada...")
        cv_scores = cross_val_score(self.model, X, encoded_labels, cv=5)
        
        # Relatório de classificação
        y_pred = self.model.predict(X)
        class_report = classification_report(
            encoded_labels, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Salvar modelo
        model_dir = self.config['paths']['models']
        save_model(self.model, self.vectorizer, self.label_encoder, model_dir)
        
        # Criar relatório
        results = {
            'training_samples': len(texts),
            'classes': list(self.label_encoder.classes_),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'config': self.config
        }
        
        # Salvar relatório
        report_path = os.path.join(self.config['paths']['logs'], 'training_report.json')
        create_training_report(results, report_path)
        
        self.logger.info(f"Treinamento concluído! Acurácia média: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results

@click.command()
@click.option('--config', default='config/config.yaml', help='Caminho para arquivo de configuração')
def main(config):
    """Executa o treinamento do classificador de documentos"""
    try:
        trainer = DocumentTrainer(config)
        results = trainer.train()
        
        print("\n" + "="*50)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*50)
        print(f"Documentos processados: {results['training_samples']}")
        print(f"Classes identificadas: {', '.join(results['classes'])}")
        print(f"Acurácia média (CV): {results['cv_mean']:.3f}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Erro durante o treinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

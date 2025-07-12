"""
Módulo de Teste do Sistema de Classificação de Documentos
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import click
import json

from common.pdf_parser import PDFParser
from common.text_preprocessor import TextPreprocessor
from common.utils import load_config, setup_logging, load_model, get_files_from_directory

class DocumentTester:
    """Classe para teste do classificador de documentos"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.pdf_parser = PDFParser()
        self.text_preprocessor = TextPreprocessor(
            self.config['models']['spacy_model']
        )
        
        # Configurar logging
        setup_logging(
            self.config['logging']['level'],
            os.path.join(self.config['paths']['logs'], 'testing.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Carregar modelo treinado
        try:
            self.model, self.vectorizer, self.label_encoder = load_model(
                self.config['paths']['models']
            )
            self.logger.info("Modelo carregado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def load_test_data(self) -> tuple:
        """
        Carrega dados de teste dos diretórios
        
        Returns:
            Tupla com textos, labels verdadeiros e nomes dos arquivos
        """
        texts = []
        true_labels = []
        filenames = []
        
        test_dir = Path(self.config['paths']['data_test'])
        
        if not test_dir.exists():
            raise FileNotFoundError(f"Diretório de teste não encontrado: {test_dir}")
        
        self.logger.info("Carregando dados de teste...")
        
        # Para cada classe de documento
        for doc_class in self.config['document_classes']:
            class_dir = test_dir / doc_class
            
            if not class_dir.exists():
                self.logger.warning(f"Diretório da classe não encontrado: {class_dir}")
                continue
            
            # Obter arquivos PDF da classe
            pdf_files = get_files_from_directory(str(class_dir), ".pdf")
            
            self.logger.info(f"Processando {len(pdf_files)} arquivos de teste da classe '{doc_class}'")
            
            # Processar cada arquivo
            for pdf_file in tqdm(pdf_files, desc=f"Testando {doc_class}"):
                # Extrair texto
                text = self.pdf_parser.extract_text(str(pdf_file))
                
                if text and len(text) >= self.config['text_processing']['min_text_length']:
                    # Pré-processar texto
                    processed_text = self.text_preprocessor.preprocess(
                        text, self.config['text_processing']
                    )
                    
                    if processed_text:
                        texts.append(processed_text)
                        true_labels.append(doc_class)
                        filenames.append(pdf_file.name)
                else:
                    self.logger.warning(f"Texto insuficiente em: {pdf_file}")
        
        self.logger.info(f"Carregados {len(texts)} documentos para teste")
        
        if len(texts) == 0:
            raise ValueError("Nenhum documento válido encontrado para teste")
        
        return texts, true_labels, filenames
    
    def predict_documents(self, texts: list) -> tuple:
        """
        Faz predições para lista de textos
        
        Args:
            texts: Lista de textos processados
            
        Returns:
            Tupla com predições e probabilidades
        """
        self.logger.info("Fazendo predições...")
        
        # Vetorizar textos
        X = self.vectorizer.transform(texts)
        
        # Fazer predições
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Converter predições para labels originais
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def evaluate_performance(self, true_labels: list, predicted_labels: list, 
                           probabilities: np.ndarray, filenames: list) -> dict:
        """
        Avalia performance do modelo
        
        Args:
            true_labels: Labels verdadeiros
            predicted_labels: Labels preditos
            probabilities: Probabilidades das predições
            filenames: Nomes dos arquivos
            
        Returns:
            Dicionário com métricas de performance
        """
        self.logger.info("Avaliando performance...")
        
        # Relatório de classificação
        class_report = classification_report(
            true_labels, predicted_labels,
            output_dict=True
        )
        
        # Matriz de confusão
        conf_matrix = confusion_matrix(
            true_labels, predicted_labels,
            labels=self.label_encoder.classes_
        )
        
        # Acurácia
        accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
        
        # Análise por arquivo
        max_probs = np.max(probabilities, axis=1)
        file_analysis = []
        
        for i, (filename, true_label, pred_label, prob) in enumerate(
            zip(filenames, true_labels, predicted_labels, max_probs)
        ):
            file_analysis.append({
                'filename': filename,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': float(prob),
                'correct': true_label == pred_label
            })
        
        # Erros de classificação
        errors = [fa for fa in file_analysis if not fa['correct']]
        
        results = {
            'total_documents': len(true_labels),
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_labels': list(self.label_encoder.classes_),
            'file_analysis': file_analysis,
            'errors': errors,
            'error_count': len(errors)
        }
        
        return results
    
    def test(self) -> dict:
        """
        Executa teste completo do modelo
        
        Returns:
            Dicionário com resultados do teste
        """
        self.logger.info("Iniciando teste do modelo...")
        
        # Carregar dados de teste
        texts, true_labels, filenames = self.load_test_data()
        
        # Fazer predições
        predicted_labels, probabilities = self.predict_documents(texts)
        
        # Avaliar performance
        results = self.evaluate_performance(
            true_labels, predicted_labels, probabilities, filenames
        )
        
        # Salvar resultados
        results_path = os.path.join(self.config['paths']['logs'], 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Teste concluído! Acurácia: {results['accuracy']:.3f}")
        self.logger.info(f"Resultados salvos em: {results_path}")
        
        return results

@click.command()
@click.option('--config', default='config/config.yaml', help='Caminho para arquivo de configuração')
@click.option('--show-errors', is_flag=True, help='Mostrar erros de classificação')
def main(config, show_errors):
    """Executa o teste do classificador de documentos"""
    try:
        tester = DocumentTester(config)
        results = tester.test()
        
        print("\n" + "="*50)
        print("TESTE CONCLUÍDO!")
        print("="*50)
        print(f"Documentos testados: {results['total_documents']}")
        print(f"Acurácia: {results['accuracy']:.3f}")
        print(f"Erros: {results['error_count']}")
        
        # Mostrar métricas por classe
        print("\nMétricas por classe:")
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"  {class_name}:")
                print(f"    Precisão: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-score: {metrics['f1-score']:.3f}")
        
        # Mostrar erros se solicitado
        if show_errors and results['errors']:
            print(f"\nErros de classificação ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  {error['filename']}: {error['true_label']} -> {error['predicted_label']} (conf: {error['confidence']:.3f})")
        
        print("="*50)
        
    except Exception as e:
        logging.error(f"Erro durante o teste: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

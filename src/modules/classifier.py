"""
Módulo de Classificação em Produção
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import numpy as np
from pathlib import Path
import click
import json

from common.pdf_parser import PDFParser
from common.text_preprocessor import TextPreprocessor
from common.utils import load_config, setup_logging, load_model

class DocumentClassifier:
    """Classe para classificação de documentos em produção"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.pdf_parser = PDFParser()
        self.text_preprocessor = TextPreprocessor(
            self.config['models']['spacy_model']
        )
        
        # Configurar logging
        setup_logging(self.config['logging']['level'])
        self.logger = logging.getLogger(__name__)
        
        # Carregar modelo treinado
        try:
            self.model, self.vectorizer, self.label_encoder = load_model(
                self.config['paths']['models']
            )
            self.logger.info("Modelo carregado com sucesso para produção")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def classify_document(self, pdf_path: str, min_confidence: float = 0.5, 
                         reject_others_threshold: float = 0.8) -> dict:
        """
        Classifica um único documento
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            min_confidence: Confiança mínima para classificação
            reject_others_threshold: Se 'outros' tiver confiança acima deste valor, rejeitar classificação
            
        Returns:
            Dicionário com resultado da classificação
        """
        try:
            self.logger.info(f"Classificando documento: {pdf_path}")
            
            # Verificar se arquivo existe
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
            
            # Extrair texto
            text = self.pdf_parser.extract_text(pdf_path)
            if not text:
                return {
                    'success': False,
                    'error': 'Não foi possível extrair texto do documento',
                    'file_path': pdf_path
                }
            
            # Verificar tamanho mínimo do texto
            if len(text) < self.config['text_processing']['min_text_length']:
                return {
                    'success': False,
                    'error': 'Texto insuficiente para classificação',
                    'file_path': pdf_path,
                    'text_length': len(text)
                }
            
            # Pré-processar texto
            processed_text = self.text_preprocessor.preprocess(
                text, self.config['text_processing']
            )
            
            if not processed_text:
                return {
                    'success': False,
                    'error': 'Texto não pôde ser processado adequadamente',
                    'file_path': pdf_path
                }
            
            # Vetorizar texto
            X = self.vectorizer.transform([processed_text])
            
            # Fazer predição
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Converter predição para label original
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            max_confidence = float(np.max(probabilities))
            
            # Criar dicionário de probabilidades por classe
            class_probabilities = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probabilities[class_name] = float(probabilities[i])
            
            # Verificar confiança mínima
            classification_accepted = max_confidence >= min_confidence
            
            # Lógica especial para categoria "outros"
            is_others = predicted_label == "outros"
            others_confidence = class_probabilities.get("outros", 0.0)
            
            # Se foi classificado como "outros" com alta confiança, considerar como "documento não reconhecido"
            if is_others and others_confidence >= reject_others_threshold:
                result = {
                    'success': True,
                    'file_path': pdf_path,
                    'predicted_class': "documento_nao_reconhecido",
                    'original_prediction': predicted_label,
                    'confidence': max_confidence,
                    'classification_accepted': False,
                    'rejection_reason': f'Documento não reconhecido (confiança "outros": {others_confidence:.3f})',
                    'min_confidence_threshold': min_confidence,
                    'reject_others_threshold': reject_others_threshold,
                    'class_probabilities': class_probabilities,
                    'text_length_original': len(text),
                    'text_length_processed': len(processed_text)
                }
            # Se não foi classificado como "outros", verificar se há competição com "outros"
            elif not is_others and others_confidence > 0.4:  # "outros" tem confiança significativa
                # Calcular margem entre a classe predita e "outros"
                margin = max_confidence - others_confidence
                
                if margin < 0.2:  # Margem pequena = incerteza
                    classification_accepted = False
                    warning_msg = f'Classificação incerta - competição com "outros" (margem: {margin:.3f})'
                else:
                    warning_msg = f'Confiança moderada - "outros" em {others_confidence:.3f}'
                
                result = {
                    'success': True,
                    'file_path': pdf_path,
                    'predicted_class': predicted_label,
                    'confidence': max_confidence,
                    'classification_accepted': classification_accepted,
                    'min_confidence_threshold': min_confidence,
                    'class_probabilities': class_probabilities,
                    'text_length_original': len(text),
                    'text_length_processed': len(processed_text),
                    'others_confidence': others_confidence,
                    'margin_vs_others': margin,
                    'warning': warning_msg
                }
            else:
                # Classificação normal
                result = {
                    'success': True,
                    'file_path': pdf_path,
                    'predicted_class': predicted_label,
                    'confidence': max_confidence,
                    'classification_accepted': classification_accepted,
                    'min_confidence_threshold': min_confidence,
                    'class_probabilities': class_probabilities,
                    'text_length_original': len(text),
                    'text_length_processed': len(processed_text)
                }
                
                if not classification_accepted:
                    result['warning'] = f'Confiança baixa ({max_confidence:.3f} < {min_confidence})'
            
            self.logger.info(f"Classificação concluída: {predicted_label} (confiança: {max_confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao classificar documento {pdf_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': pdf_path
            }
    
    def batch_classify(self, pdf_directory: str, min_confidence: float = 0.5, 
                      reject_others_threshold: float = 0.8) -> list:
        """
        Classifica múltiplos documentos de um diretório
        
        Args:
            pdf_directory: Diretório com arquivos PDF
            min_confidence: Confiança mínima para classificação
            
        Returns:
            Lista com resultados das classificações
        """
        self.logger.info(f"Classificação em lote do diretório: {pdf_directory}")
        
        directory = Path(pdf_directory)
        if not directory.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {pdf_directory}")
        
        # Encontrar arquivos PDF
        pdf_files = list(directory.glob("*.pdf"))
        self.logger.info(f"Encontrados {len(pdf_files)} arquivos PDF")
        
        results = []
        for pdf_file in pdf_files:
            result = self.classify_document(str(pdf_file), min_confidence, reject_others_threshold)
            results.append(result)
        
        return results

@click.command()
@click.option('--file', '-f', help='Caminho para arquivo PDF único')
@click.option('--directory', '-d', help='Diretório com arquivos PDF para classificação em lote')
@click.option('--config', default='config/config.yaml', help='Caminho para arquivo de configuração')
@click.option('--min-confidence', default=0.5, type=float, help='Confiança mínima para aceitar classificação')
@click.option('--reject-others', default=0.8, type=float, help='Threshold para rejeitar documentos classificados como "outros"')
@click.option('--output', '-o', help='Arquivo de saída para resultados (JSON)')
@click.option('--verbose', is_flag=True, help='Saída detalhada')
def main(file, directory, config, min_confidence, reject_others, output, verbose):
    """Classifica documentos PDF usando modelo treinado"""
    
    if not file and not directory:
        click.echo("❌ Erro: Especifique --file ou --directory")
        sys.exit(1)
    
    if file and directory:
        click.echo("❌ Erro: Especifique apenas --file OU --directory, não ambos")
        sys.exit(1)
    
    try:
        classifier = DocumentClassifier(config)
        
        if file:
            # Classificação de arquivo único
            result = classifier.classify_document(file, min_confidence, reject_others)
            results = [result]
        else:
            # Classificação em lote
            results = classifier.batch_classify(directory, min_confidence, reject_others)
        
        # Salvar resultados se solicitado
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            click.echo(f"📄 Resultados salvos em: {output}")
        
        # Exibir resultados
        click.echo("\n" + "="*60)
        click.echo("RESULTADOS DA CLASSIFICAÇÃO")
        click.echo("="*60)
        
        successful = 0
        total = len(results)
        
        for result in results:
            filename = Path(result['file_path']).name
            
            if result['success']:
                successful += 1
                
                # Determinar status visual
                if result['predicted_class'] == "documento_nao_reconhecido":
                    status = "🚫"
                    class_display = "Documento NÃO RECONHECIDO"
                    confidence = result['confidence']
                elif result['classification_accepted']:
                    status = "✅"
                    class_display = result['predicted_class'].replace('_', ' ').title()
                    confidence = result['confidence']
                else:
                    status = "⚠️"
                    class_display = result['predicted_class'].replace('_', ' ').title()
                    confidence = result['confidence']
                
                click.echo(f"{status} {filename}")
                click.echo(f"   Classe: {class_display}")
                click.echo(f"   Confiança: {confidence:.3f}")
                
                # Mostrar avisos específicos
                if 'rejection_reason' in result:
                    click.echo(f"   🚫 {result['rejection_reason']}")
                elif 'warning' in result:
                    click.echo(f"   ⚠️  {result['warning']}")
                
                # Informações extras sobre competição com "outros"
                if 'margin_vs_others' in result:
                    click.echo(f"   📊 Margem vs 'outros': {result['margin_vs_others']:.3f}")
                
                if verbose:
                    click.echo("   Probabilidades por classe:")
                    for class_name, prob in result['class_probabilities'].items():
                        icon = "🎯" if class_name == result.get('original_prediction', result['predicted_class']) else "  "
                        click.echo(f"     {icon} {class_name}: {prob:.3f}")
                
                click.echo()
            else:
                click.echo(f"❌ {filename}")
                click.echo(f"   Erro: {result['error']}")
                click.echo()
        
        click.echo("="*60)
        click.echo(f"Documentos processados: {total}")
        click.echo(f"Classificações bem-sucedidas: {successful}")
        click.echo(f"Taxa de sucesso: {successful/total*100:.1f}%")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

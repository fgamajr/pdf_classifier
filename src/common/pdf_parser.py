"""
Módulo para extração de texto de arquivos PDF
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PDFParser:
    """Classe para extrair texto de arquivos PDF"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
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
                logger.error(f"Arquivo não encontrado: {pdf_path}")
                return None
                
            if pdf_path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Extensão não suportada: {pdf_path.suffix}")
                return None
            
            # Abrir documento PDF
            doc = fitz.open(str(pdf_path))
            text = ""
            
            # Extrair texto de todas as páginas
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                
            doc.close()
            
            if not text.strip():
                logger.warning(f"Nenhum texto encontrado em: {pdf_path}")
                return None
                
            logger.info(f"Texto extraído com sucesso de: {pdf_path}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {pdf_path}: {str(e)}")
            return None
    
    def extract_metadata(self, pdf_path: str) -> dict:
        """
        Extrai metadados do PDF
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            
        Returns:
            Dicionário com metadados
        """
        try:
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Erro ao extrair metadados de {pdf_path}: {str(e)}")
            return {}

"""
Utilitários gerais do sistema
"""
import yaml
import json
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Carrega arquivo de configuração YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Erro ao carregar configuração: {e}")
        return {}

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configura o sistema de logging"""
    level = getattr(logging, log_level.upper())
    
    # Formato do log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configurar logger root
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    # Handler para arquivo se especificado
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def save_model(model, vectorizer, label_encoder, model_dir: str) -> None:
    """Salva modelo treinado e componentes"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar componentes
    joblib.dump(model, model_dir / "classifier.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")
    
    logging.info(f"Modelo salvo em: {model_dir}")

def load_model(model_dir: str) -> tuple:
    """Carrega modelo treinado e componentes"""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Diretório do modelo não encontrado: {model_dir}")
    
    # Carregar componentes
    model = joblib.load(model_dir / "classifier.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    label_encoder = joblib.load(model_dir / "label_encoder.pkl")
    
    logging.info(f"Modelo carregado de: {model_dir}")
    return model, vectorizer, label_encoder

def get_files_from_directory(directory: str, extension: str = ".pdf") -> List[Path]:
    """Obtém lista de arquivos de um diretório"""
    directory = Path(directory)
    if not directory.exists():
        logging.warning(f"Diretório não encontrado: {directory}")
        return []
    
    files = list(directory.rglob(f"*{extension}"))
    logging.info(f"Encontrados {len(files)} arquivos em {directory}")
    return files

def create_training_report(results: Dict[str, Any], output_path: str) -> None:
    """Cria relatório de treinamento"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Relatório salvo em: {output_path}")

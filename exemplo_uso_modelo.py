"""
Exemplos Práticos de Uso do Modelo Treinado
===========================================

Este arquivo mostra diferentes formas de usar o modelo
de classificação de documentos em outros projetos.
"""

# ============================================
# EXEMPLO 1: Uso Básico em Qualquer Projeto
# ============================================

from portable_document_classifier import DocumentClassifierPortable

def exemplo_basico():
    """Uso mais simples possível"""
    
    # Inicializar com o caminho dos modelos
    classifier = DocumentClassifierPortable("path/to/models/")
    
    # Classificar um PDF
    resultado = classifier.classify_pdf("documento.pdf")
    
    if resultado['success']:
        print(f"Documento é: {resultado['predicted_class']}")
        print(f"Confiança: {resultado['confidence']:.1%}")
    else:
        print(f"Erro: {resultado['error']}")

# ============================================
# EXEMPLO 2: API Web com Flask
# ============================================

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Inicializar classificador uma vez quando a API subir
CLASSIFIER = DocumentClassifierPortable("models/")

@app.route('/classify', methods=['POST'])
def api_classify():
    """API endpoint para classificar documentos"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Arquivo vazio'}), 400
    
    # Salvar arquivo temporariamente
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    try:
        # Classificar
        resultado = CLASSIFIER.classify_pdf(temp_path)
        
        # Limpar arquivo temporário
        os.remove(temp_path)
        
        return jsonify(resultado)
        
    except Exception as e:
        # Limpar arquivo em caso de erro
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def api_info():
    """Endpoint para informações do modelo"""
    return jsonify(CLASSIFIER.get_model_info())

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# ============================================
# EXEMPLO 3: Processamento em Lote
# ============================================

import os
import csv
from pathlib import Path

def processar_diretorio_completo(diretorio_pdfs, arquivo_resultado):
    """
    Processa todos os PDFs de um diretório e salva resultados em CSV
    """
    
    classifier = DocumentClassifierPortable("models/")
    
    # Processar todos os PDFs
    resultados = classifier.batch_classify(diretorio_pdfs)
    
    # Salvar em CSV
    with open(arquivo_resultado, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'arquivo', 'classe_predita', 'confianca', 
            'aceito', 'erro', 'tamanho_texto'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for resultado in resultados:
            if resultado['success']:
                writer.writerow({
                    'arquivo': Path(resultado['file_path']).name,
                    'classe_predita': resultado['predicted_class'],
                    'confianca': f"{resultado['confidence']:.3f}",
                    'aceito': resultado['classification_accepted'],
                    'erro': '',
                    'tamanho_texto': resultado['text_length_original']
                })
            else:
                writer.writerow({
                    'arquivo': Path(resultado['file_path']).name,
                    'classe_predita': '',
                    'confianca': '',
                    'aceito': False,
                    'erro': resultado['error'],
                    'tamanho_texto': 0
                })
    
    print(f"Resultados salvos em: {arquivo_resultado}")

# ============================================
# EXEMPLO 4: Integração com Sistema de Arquivos
# ============================================

import shutil

def organizar_documentos_por_classe(diretorio_entrada, diretorio_saida):
    """
    Classifica documentos e os organiza em pastas por categoria
    """
    
    classifier = DocumentClassifierPortable("models/")
    
    # Criar diretórios de saída
    Path(diretorio_saida).mkdir(exist_ok=True)
    
    # Processar documentos
    resultados = classifier.batch_classify(diretorio_entrada)
    
    estatisticas = {'total': 0, 'processados': 0, 'rejeitados': 0}
    
    for resultado in resultados:
        estatisticas['total'] += 1
        
        if resultado['success'] and resultado['classification_accepted']:
            # Criar pasta da classe se não existir
            classe = resultado['predicted_class']
            pasta_classe = Path(diretorio_saida) / classe
            pasta_classe.mkdir(exist_ok=True)
            
            # Copiar arquivo para pasta da classe
            origem = resultado['file_path']
            destino = pasta_classe / Path(origem).name
            shutil.copy2(origem, destino)
            
            estatisticas['processados'] += 1
            print(f"✅ {Path(origem).name} → {classe}/")
            
        else:
            # Criar pasta para rejeitados
            pasta_rejeitados = Path(diretorio_saida) / "rejeitados"
            pasta_rejeitados.mkdir(exist_ok=True)
            
            origem = resultado['file_path']
            destino = pasta_rejeitados / Path(origem).name
            shutil.copy2(origem, destino)
            
            estatisticas['rejeitados'] += 1
            print(f"❌ {Path(origem).name} → rejeitados/")
    
    print(f"\nEstatísticas:")
    print(f"Total: {estatisticas['total']}")
    print(f"Processados: {estatisticas['processados']}")
    print(f"Rejeitados: {estatisticas['rejeitados']}")

# ============================================
# EXEMPLO 5: Monitoramento de Pasta
# ============================================

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentHandler(FileSystemEventHandler):
    """Handler para monitorar novos documentos"""
    
    def __init__(self, models_path):
        self.classifier = DocumentClassifierPortable(models_path)
    
    def on_created(self, event):
        """Executado quando um novo arquivo é criado"""
        
        if event.is_directory:
            return
        
        if not event.src_path.endswith('.pdf'):
            return
        
        print(f"📄 Novo documento detectado: {event.src_path}")
        
        # Aguardar um pouco (arquivo pode estar sendo copiado)
        time.sleep(2)
        
        # Classificar
        resultado = self.classifier.classify_pdf(event.src_path)
        
        if resultado['success'] and resultado['classification_accepted']:
            classe = resultado['predicted_class']
            confianca = resultado['confidence']
            print(f"🎯 Classificado como: {classe} (confiança: {confianca:.3f})")
            
            # Aqui você pode adicionar ações automáticas:
            # - Mover para pasta específica
            # - Enviar notificação
            # - Salvar em banco de dados
            # - etc.
        else:
            print(f"❌ Não foi possível classificar: {resultado.get('error', 'Confiança baixa')}")

def monitorar_pasta(pasta_monitorada, models_path):
    """
    Monitora uma pasta e classifica automaticamente novos PDFs
    """
    
    event_handler = DocumentHandler(models_path)
    observer = Observer()
    observer.schedule(event_handler, pasta_monitorada, recursive=False)
    
    print(f"🔍 Monitorando pasta: {pasta_monitorada}")
    print("Pressione Ctrl+C para parar...")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Monitoramento interrompido.")
    
    observer.join()

# ============================================
# EXEMPLO 6: Linha de Comando Simples
# ============================================

import argparse

def main():
    """Interface de linha de comando"""
    
    parser = argparse.ArgumentParser(description='Classificador de Documentos')
    parser.add_argument('--models', required=True, help='Diretório dos modelos')
    parser.add_argument('--file', help='Arquivo PDF para classificar')
    parser.add_argument('--directory', help='Diretório com PDFs para processar')
    parser.add_argument('--organize', help='Organizar documentos por classe')
    parser.add_argument('--monitor', help='Monitorar pasta para novos documentos')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confiança mínima')
    
    args = parser.parse_args()
    
    if args.file:
        # Classificar arquivo único
        classifier = DocumentClassifierPortable(args.models)
        resultado = classifier.classify_pdf(args.file, args.confidence)
        
        if resultado['success']:
            print(f"Classe: {resultado['predicted_class']}")
            print(f"Confiança: {resultado['confidence']:.3f}")
        else:
            print(f"Erro: {resultado['error']}")
    
    elif args.directory:
        # Processar diretório
        processar_diretorio_completo(args.directory, "resultados.csv")
    
    elif args.organize:
        # Organizar documentos
        organizar_documentos_por_classe(args.organize, "documentos_organizados")
    
    elif args.monitor:
        # Monitorar pasta
        monitorar_pasta(args.monitor, args.models)
    
    else:
        parser.print_help()

# ============================================
# EXEMPLO 7: Integração com Banco de Dados
# ============================================

import sqlite3
from datetime import datetime

def salvar_resultado_bd(resultado, conexao_bd):
    """Salva resultado de classificação no banco de dados"""
    
    cursor = conexao_bd.cursor()
    
    # Criar tabela se não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classificacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arquivo TEXT NOT NULL,
            classe_predita TEXT,
            confianca REAL,
            aceito BOOLEAN,
            erro TEXT,
            data_processamento TIMESTAMP,
            tamanho_texto INTEGER
        )
    ''')
    
    # Inserir resultado
    cursor.execute('''
        INSERT INTO classificacoes 
        (arquivo, classe_predita, confianca, aceito, erro, data_processamento, tamanho_texto)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        resultado.get('file_path', ''),
        resultado.get('predicted_class', ''),
        resultado.get('confidence', 0.0),
        resultado.get('classification_accepted', False),
        resultado.get('error', ''),
        datetime.now(),
        resultado.get('text_length_original', 0)
    ))
    
    conexao_bd.commit()

def processar_com_bd(diretorio_pdfs, caminho_bd):
    """Processa PDFs e salva resultados em banco SQLite"""
    
    classifier = DocumentClassifierPortable("models/")
    
    # Conectar ao banco
    conn = sqlite3.connect(caminho_bd)
    
    try:
        resultados = classifier.batch_classify(diretorio_pdfs)
        
        for resultado in resultados:
            salvar_resultado_bd(resultado, conn)
            print(f"💾 Salvo no BD: {Path(resultado['file_path']).name}")
        
        print(f"✅ {len(resultados)} resultados salvos em {caminho_bd}")
        
    finally:
        conn.close()

# ============================================
# EXEMPLO DE USO DOS EXEMPLOS 😄
# ============================================

if __name__ == "__main__":
    
    print("🎯 Exemplos de uso do modelo treinado:")
    print()
    print("1. Uso básico:")
    print("   exemplo_basico()")
    print()
    print("2. Processar diretório:")
    print("   processar_diretorio_completo('pdfs/', 'resultados.csv')")
    print()
    print("3. Organizar por categoria:")
    print("   organizar_documentos_por_classe('entrada/', 'saida/')")
    print()
    print("4. Monitorar pasta:")
    print("   monitorar_pasta('pasta_monitorada/', 'models/')")
    print()
    print("5. Linha de comando:")
    print("   python exemplos_uso_modelo.py --models models/ --file documento.pdf")
    print()
    print("6. Banco de dados:")
    print("   processar_com_bd('pdfs/', 'classificacoes.db')")
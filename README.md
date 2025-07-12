# Sistema de Classificação de Documentos

Sistema inteligente para classificação automática de documentos PDF usando Machine Learning.

## 🎯 Objetivo

Classificar automaticamente documentos PDF em categorias como:
- **Diplomas** - Certificados de graduação, pós-graduação
- **Comprovantes de Renda** - Holerites, declarações de renda
- **Comprovantes de Endereço** - Contas de luz, água, telefone
- **Outros** - Documentos que não se encaixam nas categorias acima (contratos, notas fiscais, etc.)

⚠️ **Importante**: A categoria "outros" ajuda o sistema a **NÃO** classificar incorretamente documentos que estão fora do escopo esperado.

## 🏗️ Arquitetura

O sistema é dividido em 3 módulos principais:

1. **Trainer** (`src/modules/trainer.py`) - Treina o modelo com documentos de exemplo
2. **Tester** (`src/modules/tester.py`) - Testa e valida o modelo treinado
3. **Classifier** (`src/modules/classifier.py`) - Classifica novos documentos (produção)
4. **Portable Classifier** (`portable_document_classifier.py`) - Módulo portável para reutilização em outros projetos

## 📁 Estrutura do Projeto

```
document_classifier/
├── src/
│   ├── common/           # Funções compartilhadas
│   │   ├── pdf_parser.py
│   │   ├── text_preprocessor.py
│   │   └── utils.py
│   └── modules/          # Módulos principais
│       ├── trainer.py
│       ├── tester.py
│       └── classifier.py
├── data/
│   ├── train/           # Documentos para treinamento
│   │   ├── diplomas/
│   │   ├── comprovantes_renda/
│   │   ├── comprovantes_endereco/
│   │   └── outros/      # ⭐ Categoria importante!
│   ├── test/            # Documentos para teste
│   └── production/      # Novos documentos para classificar
├── models/              # 🧠 Modelos treinados (SEU TESOURO!)
│   ├── classifier.pkl   # Modelo de IA treinado
│   ├── vectorizer.pkl   # Dicionário de palavras
│   └── label_encoder.pkl # Tradutor de classes
├── logs/                # Logs do sistema
├── config/              # Configurações
├── venv/                # Ambiente virtual Python
├── portable_document_classifier.py  # 🚀 Módulo portável
├── exemplos_uso_modelo.py           # 📋 Exemplos práticos
└── README.md
```

## 🚀 Como Usar

### 1. Preparar Dados de Treinamento

Coloque seus documentos PDF nas pastas correspondentes em `data/train/`:
- `data/train/diplomas/` - Documentos de diploma e certificados
- `data/train/comprovantes_renda/` - Comprovantes de renda e holerites  
- `data/train/comprovantes_endereco/` - Comprovantes de endereço
- `data/train/outros/` - **IMPORTANTE**: Outros tipos de documentos (contratos, notas fiscais, receitas médicas, etc.)

💡 **Dica**: A pasta "outros" é crucial! Inclua documentos variados que NÃO são das categorias principais. Isso evita classificações incorretas.

### 2. Treinar o Modelo

```bash
./run_training.sh
```

Ou manualmente:
```bash
source venv/bin/activate
cd src/modules
python trainer.py
```

### 3. Testar o Modelo

```bash
./run_testing.sh
```

### 4. Classificar Novos Documentos

```bash
./run_classifier.sh /caminho/para/documento.pdf
```

## 📊 Comandos Detalhados

### Treinamento
```bash
# Treinamento básico
python trainer.py

# Com configuração customizada
python trainer.py --config config/config.yaml
```

### Teste
```bash
# Teste básico
python tester.py

# Mostrar erros de classificação
python tester.py --show-errors
```

### Classificação
```bash
# Classificar arquivo único
python classifier.py --file documento.pdf

# Classificação em lote
python classifier.py --directory /pasta/com/pdfs/

# Com saída detalhada
python classifier.py --file documento.pdf --verbose

# Salvar resultados em arquivo
python classifier.py --directory /pasta/ --output resultados.json

# Definir confiança mínima
python classifier.py --file documento.pdf --min-confidence 0.7

# Ajustar threshold para rejeição de "outros"
python classifier.py --file documento.pdf --reject-others 0.9
```

## 🎯 Categoria "Outros" - Por que é Importante?

A categoria "outros" é fundamental para um sistema robusto:

### ✅ **Com categoria "outros":**
- ❌ Contrato → Classificado como "outros" → Sistema rejeita ✅
- ❌ Nota fiscal → Classificado como "outros" → Sistema rejeita ✅  
- ✅ Diploma → Classificado como "diploma" → Sistema aceita ✅

### ❌ **Sem categoria "outros":**
- ❌ Contrato → **Forçado** como "diploma" → Erro grave! ❌
- ❌ Nota fiscal → **Forçado** como "comprovante_renda" → Erro grave! ❌

### 📁 **Exemplos para pasta "outros":**
- Contratos diversos
- Notas fiscais  
- Receitas médicas
- Certidões
- Extratos bancários
- Declarações
- Atas de reunião

## 🚀 Portabilidade e Reutilização

### 📦 Onde o Modelo Fica Salvo

Após o treinamento, os modelos são salvos em `models/`:
```
models/
├── classifier.pkl      # 🧠 O cérebro da IA (2-50MB)
├── vectorizer.pkl      # 📝 Dicionário de palavras (1-10MB)
├── label_encoder.pkl   # 🏷️ Tradutor de classes (pequeno)
└── training_report.json # 📊 Relatório opcional
```

### 🔄 Como Reutilizar em Outros Projetos

#### Opção 1: Uso Simples (Copiar Modelos)
```bash
# Copie a pasta models/ para seu novo projeto
cp -r document_classifier/models/ /meu_novo_projeto/

# Use com joblib
import joblib
model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
```

#### Opção 2: Módulo Portável (Recomendado)
```python
# Copie portable_document_classifier.py para seu projeto
from portable_document_classifier import DocumentClassifierPortable

# Uso em 2 linhas!
classifier = DocumentClassifierPortable("models/")
resultado = classifier.classify_pdf("documento.pdf")

print(f"Documento é: {resultado['predicted_class']}")
print(f"Confiança: {resultado['confidence']:.1%}")
```

### 🎯 Exemplos Práticos de Reutilização

#### 1. Classificação Básica
```python
from portable_document_classifier import DocumentClassifierPortable

classifier = DocumentClassifierPortable("models/")
resultado = classifier.classify_pdf("documento.pdf")

if resultado['success'] and resultado['classification_accepted']:
    print(f"✅ Documento: {resultado['predicted_class']}")
else:
    print(f"❌ Rejeitado: {resultado.get('error', 'Confiança baixa')}")
```

#### 2. API REST com Flask
```python
from flask import Flask, request, jsonify
from portable_document_classifier import DocumentClassifierPortable

app = Flask(__name__)
classifier = DocumentClassifierPortable("models/")

@app.route('/classify', methods=['POST'])
def classify_document():
    file = request.files['file']
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    resultado = classifier.classify_pdf(temp_path)
    os.remove(temp_path)
    
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(port=5000)
```

#### 3. Processamento em Lote
```python
# Classificar todos os PDFs de uma pasta
resultados = classifier.batch_classify("pasta_com_pdfs/")

for resultado in resultados:
    if resultado['success']:
        arquivo = Path(resultado['file_path']).name
        classe = resultado['predicted_class']
        print(f"{arquivo} → {classe}")
```

#### 4. Organização Automática
```python
def organizar_documentos(pasta_entrada, pasta_saida):
    """Classifica e organiza documentos em pastas por categoria"""
    
    classifier = DocumentClassifierPortable("models/")
    resultados = classifier.batch_classify(pasta_entrada)
    
    for resultado in resultados:
        if resultado['success'] and resultado['classification_accepted']:
            # Criar pasta da categoria
            categoria = resultado['predicted_class']
            pasta_categoria = Path(pasta_saida) / categoria
            pasta_categoria.mkdir(exist_ok=True)
            
            # Mover arquivo
            origem = resultado['file_path']
            destino = pasta_categoria / Path(origem).name
            shutil.copy2(origem, destino)
            
            print(f"📁 {Path(origem).name} → {categoria}/")
```

#### 5. Monitoramento de Pasta
```python
# Monitora pasta e classifica automaticamente novos PDFs
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DocumentHandler(FileSystemEventHandler):
    def __init__(self):
        self.classifier = DocumentClassifierPortable("models/")
    
    def on_created(self, event):
        if event.src_path.endswith('.pdf'):
            resultado = self.classifier.classify_pdf(event.src_path)
            print(f"📄 Novo documento: {resultado['predicted_class']}")

# Usar
observer = Observer()
observer.schedule(DocumentHandler(), "pasta_monitorada/", recursive=False)
observer.start()
```

### 💡 Vantagens do Sistema Portável

- ✅ **Portabilidade Total**: Funciona em qualquer projeto Python
- ✅ **Performance**: Carregamento 1-2s, classificação 0.1-0.5s por PDF
- ✅ **Flexibilidade**: Ajustar confiança, processar lotes, integrar APIs
- ✅ **Memória Eficiente**: 50-200MB RAM
- ✅ **Casos de Uso**: RH, bancos, cartórios, APIs de terceiros

### 📋 O que Levar para Outro Projeto

**Arquivos Obrigatórios:**
- ✅ `models/classifier.pkl`
- ✅ `models/vectorizer.pkl`
- ✅ `models/label_encoder.pkl`
- ✅ `portable_document_classifier.py`

**Dependências:**
```bash
pip install PyMuPDF scikit-learn spacy joblib
python -m spacy download pt_core_news_sm
```

## ⚙️ Configuração

Edite `config/config.yaml` para ajustar:
- Tipo de classificador (naive_bayes, svm, random_forest)
- Parâmetros de processamento de texto
- Caminhos de diretórios
- Configurações de logging
- Thresholds de confiança

## 📈 Casos de Uso Avançados

### 💼 Sistema de RH
```python
# Classificar currículos e documentos automaticamente
def processar_candidato(pasta_candidato):
    resultados = classifier.batch_classify(pasta_candidato)
    
    documentos_validos = []
    for r in resultados:
        if r['classification_accepted']:
            documentos_validos.append({
                'tipo': r['predicted_class'],
                'arquivo': r['file_path'],
                'confianca': r['confidence']
            })
    
    return documentos_validos
```

### 🏦 Sistema Bancário
```python
# Validar documentos de abertura de conta
def validar_documentos_conta(documentos_cliente):
    tipos_necessarios = ['comprovantes_renda', 'comprovantes_endereco']
    tipos_encontrados = []
    
    for doc in documentos_cliente:
        resultado = classifier.classify_pdf(doc)
        if resultado['classification_accepted']:
            tipos_encontrados.append(resultado['predicted_class'])
    
    documentacao_completa = all(tipo in tipos_encontrados for tipo in tipos_necessarios)
    return documentacao_completa
```

### 📊 Análise e Relatórios
```python
# Gerar estatísticas de documentos processados
def gerar_relatorio(pasta_documentos):
    resultados = classifier.batch_classify(pasta_documentos)
    
    estatisticas = {
        'total': len(resultados),
        'aceitos': 0,
        'rejeitados': 0,
        'por_categoria': {}
    }
    
    for r in resultados:
        if r['success'] and r['classification_accepted']:
            estatisticas['aceitos'] += 1
            categoria = r['predicted_class']
            estatisticas['por_categoria'][categoria] = estatisticas['por_categoria'].get(categoria, 0) + 1
        else:
            estatisticas['rejeitados'] += 1
    
    return estatisticas
```

## 🔧 Dependências

### Dependências Básicas (Sempre Necessárias)
- Python 3.8+
- PyMuPDF (extração de texto PDF)
- scikit-learn (machine learning)
- spaCy (processamento de linguagem natural)
- NLTK (processamento de texto)
- joblib (serialização de modelos)

```bash
# Instalar dependências básicas
pip install -r requirements.txt
```

### Dependências Opcionais (Para Funcionalidades Avançadas)

Para usar os exemplos avançados, instale as dependências opcionais:

```txt
# requirements-optional.txt
# Instalar apenas se for usar as funcionalidades avançadas

watchdog==3.0.0        # Monitoramento de pastas
flask==3.0.0           # API REST
flask-cors==4.0.0      # CORS para APIs
sqlalchemy==2.0.23     # Banco de dados
streamlit==1.28.1      # Interface web (futuro)
Pillow==10.1.0         # Processamento de imagens (OCR futuro)
```

### 🚀 Como Instalar

```bash
# Dependências básicas (sempre necessárias)
pip install -r requirements.txt

# Dependências opcionais (só se for usar)
pip install -r requirements-optional.txt

# Ou instalar apenas o que precisa:
pip install watchdog flask  # Para monitoramento + API
pip install streamlit       # Para interface web
pip install sqlalchemy     # Para banco de dados
```

### 💡 Dica Pro
Mantenha separado para não "inchar" a instalação básica. Usuário instala só o que vai usar! 🎯

**Funcionalidades por dependência:**
- `watchdog` → Monitoramento automático de pastas
- `flask` → Criar APIs REST
- `sqlalchemy` → Salvar resultados em banco de dados
- `streamlit` → Interface web (desenvolvimento futuro)
- `Pillow` → Processamento de imagens para OCR

## 📝 Logs

Logs são salvos em:
- `logs/training.log` - Logs de treinamento
- `logs/testing.log` - Logs de teste
- `logs/training_report.json` - Relatório de treinamento
- `logs/test_results.json` - Resultados de teste

## 🐛 Solução de Problemas

### Erro: "Modelo spaCy não encontrado"
```bash
python -m spacy download pt_core_news_sm
```

### Erro: "NLTK data not found"
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Erro: "Arquivo de modelo não encontrado"
```bash
# Verifique se os arquivos existem
ls -la models/
# Deve mostrar: classifier.pkl, vectorizer.pkl, label_encoder.pkl
```

### Baixa acurácia
- Adicione mais documentos de treinamento (recomendado: 20-50 por categoria)
- Inclua documentos variados na pasta "outros"
- Verifique qualidade dos PDFs (texto extraível)
- Ajuste parâmetros no `config.yaml`
- Use `--show-errors` no teste para identificar padrões

### Performance lenta
- Reduza `vectorizer_max_features` no config
- Use modelo `naive_bayes` em vez de `svm`
- Processe documentos em lotes menores

## 🚀 Deploy e Produção

### Docker (Futuro)
```dockerfile
# Exemplo de Dockerfile para produção
FROM python:3.9-slim
COPY models/ /app/models/
COPY portable_document_classifier.py /app/
RUN pip install PyMuPDF scikit-learn spacy joblib
RUN python -m spacy download pt_core_news_sm
CMD ["python", "/app/api.py"]
```

### Considerações de Segurança
- ⚠️ **Dados Sensíveis**: PDFs podem conter informações pessoais
- ⚠️ **Validação**: Sempre validar entrada de PDFs
- ⚠️ **Logs**: Não logar conteúdo dos documentos
- ⚠️ **Acesso**: Controlar acesso aos modelos treinados

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `logs/` para mais detalhes
2. Teste com documentos de exemplo primeiro
3. Valide que os 3 arquivos de modelo existem em `models/`
4. Confirme que as dependências estão instaladas

## 🎯 Próximos Passos

Após dominar o básico, considere:
- [ ] Treinar com mais tipos de documentos
- [ ] Implementar OCR para documentos digitalizados
- [ ] Criar interface web com Streamlit/Flask
- [ ] Adicionar validação de conteúdo (não apenas classificação)
- [ ] Integrar com sistemas existentes via API
- [ ] Implementar versionamento de modelos

---

**🚀 O sistema está pronto para uso profissional e pode ser facilmente integrado em qualquer projeto Python!**
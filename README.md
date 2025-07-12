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
│   ├── test/            # Documentos para teste
│   └── production/      # Novos documentos para classificar
├── models/              # Modelos treinados
├── logs/                # Logs do sistema
├── config/              # Configurações
└── venv/                # Ambiente virtual Python
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

## ⚙️ Configuração

Edite `config/config.yaml` para ajustar:
- Tipo de classificador (naive_bayes, svm, random_forest)
- Parâmetros de processamento de texto
- Caminhos de diretórios
- Configurações de logging

## 📈 Melhorias Futuras

- [ ] Suporte a mais tipos de documentos
- [ ] Interface web
- [ ] API REST
- [ ] Suporte a documentos digitalizados (OCR)
- [ ] Modelos deep learning
- [ ] Deploy com Docker

## 🔧 Dependências

- Python 3.8+
- PyMuPDF (extração de texto PDF)
- scikit-learn (machine learning)
- spaCy (processamento de linguagem natural)
- NLTK (processamento de texto)

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

### Baixa acurácia
- Adicione mais documentos de treinamento
- Verifique qualidade dos PDFs (texto extraível)
- Ajuste parâmetros no config.yaml

## 📞 Suporte

Para dúvidas ou problemas, verifique os logs em `logs/` para mais detalhes.

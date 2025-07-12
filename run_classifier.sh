#!/bin/bash
echo "🔍 Classificando documento..."
if [ -z "$1" ]; then
    echo "Uso: ./run_classifier.sh /caminho/para/documento.pdf"
    exit 1
fi
source venv/bin/activate
cd src/modules
python classifier.py --file "$1" --verbose

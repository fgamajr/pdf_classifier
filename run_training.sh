#!/bin/bash
echo "🚀 Iniciando treinamento do modelo..."
source venv/bin/activate
cd src/modules
python trainer.py
echo "✅ Treinamento concluído!"

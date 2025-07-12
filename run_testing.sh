#!/bin/bash
echo "🧪 Iniciando teste do modelo..."
source venv/bin/activate
cd src/modules
python tester.py --show-errors
echo "✅ Teste concluído!"

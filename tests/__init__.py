import os

# os testes usam os valores padrão, não a calibração real em pesos/calibracao.json
os.environ["CALIBRACAO"] = os.devnull

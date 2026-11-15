#!/usr/bin/env python3
"""Calcula o tamanho da amostra e sorteia a AAS (amostragem aleatória simples, sem reposição).

Lê  app/dados/populacao.parquet  (a população já limpa, 4.250.945 movimentações)
Grava:
  dados/amostra/amostra_aas.csv     a amostra (uma linha por movimentação sorteada, com id_populacao)
  dados/amostra/amostra_meta.json   parâmetros, semente, hash da população e do código: documenta o procedimento
  app/dados/amostra_aas.csv, app/dados/amostra_meta.json   cópias lidas pelo app

Procedimento (a conta mora em app/amostragem.py, o mesmo código que o app usa):
  1. n0 = z^2 * p * (1 - p) / e^2          (tamanho para estimar uma proporção)
  2. n  = ceil(n0)                          (a correção para população finita só entra se n0 passar de 5% de N)
  3. sorteio: numpy.random.default_rng(SEMENTE).choice(N, size=n, replace=False)
     (cada subconjunto de n movimentações tem a mesma chance; `id_populacao` é a posição da linha na população)

n é SEMPRE calculado aqui a partir de N, confiança, erro e p: se a população mudar (outro mês, outras regras
de limpeza), basta rodar o script de novo. Não há tamanho digitado à mão.

Reprodutível: com o mesmo arquivo de população e a mesma semente, o sorteio dá exatamente a mesma amostra.

Uso:  .venv/bin/python scripts/sorteio_amostra.py
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "app"))
from amostragem import indices_aas, tamanho_amostra  # noqa: E402

POP = RAIZ / "app" / "dados" / "populacao.parquet"
DIR_AMOSTRA = RAIZ / "dados" / "amostra"
DIR_APP = RAIZ / "app" / "dados"

# ---- parâmetros do dimensionamento (justificativas em amostra_meta.json e na aba Amostra) ----
CONFIANCA = 0.95   # nível de confiança usual em pesquisas
ERRO = 0.02        # margem de erro de 2 pontos percentuais
P = 0.5            # proporção esperada: 0,5 é o caso mais conservador (maior variância p(1-p))
SEMENTE = 20260929  # fixa o sorteio; escolhida arbitrariamente, não depende dos dados


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    pop = pd.read_parquet(POP)
    N = len(pop)

    t = tamanho_amostra(CONFIANCA, ERRO, P, N)
    z, n0, n = t["z"], t["n0"], t["n"]
    idx = indices_aas(N, n, SEMENTE)
    amostra = pop.iloc[idx].copy()
    for c in amostra.select_dtypes("category"):
        amostra[c] = amostra[c].astype(str)
    amostra.insert(0, "id_populacao", idx)

    meta = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "populacao_N": int(N),
        "arquivo_populacao": "app/dados/populacao.parquet",
        "sha256_populacao": sha256(POP),
        "confianca": CONFIANCA, "z": round(z, 6), "margem_erro": ERRO, "p": P,
        "n_formula_sem_arredondar": round(n0, 4), "n": n, "correcao_populacao_finita_aplicada": t["usa_correcao"], "n_se_corrigisse_populacao_finita": t["n_com_correcao"],
        "fracao_amostral_pct": round(100 * n / N, 6),
        "metodo": "AAS sem reposição: numpy.random.default_rng(semente).choice(N, n, replace=False)",
        "calculadora_conferencia": "calculadora de tamanho de amostra da USP/Bauru (conferir manualmente)",
        "semente": SEMENTE, "numpy": np.__version__,
        "justificativas": {
            "confianca": "95% é o padrão em pesquisas; equilibra segurança e tamanho da amostra.",
            "margem_erro": f"{ERRO * 100:.0f} pontos percentuais: precisão suficiente para comparar proporções com a população.",
            "p": "0,5 maximiza p(1-p), então garante a margem de erro para qualquer proporção que se queira estimar.",
        },
    }

    DIR_AMOSTRA.mkdir(parents=True, exist_ok=True)
    amostra.to_csv(DIR_AMOSTRA / "amostra_aas.csv", index=False)
    (DIR_AMOSTRA / "amostra_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy(DIR_AMOSTRA / "amostra_aas.csv", DIR_APP / "amostra_aas.csv")
    shutil.copy(DIR_AMOSTRA / "amostra_meta.json", DIR_APP / "amostra_meta.json")

    # conferências
    rel = pd.read_csv(DIR_AMOSTRA / "amostra_aas.csv")
    assert len(rel) == n and rel["id_populacao"].is_unique
    print(f"z = {z:.4f}; n0 = {n0:.2f} -> n = {n} (correção p/ população finita aplicada: {t['usa_correcao']}); N = {N:,}")
    print(f"amostra: {len(rel)} linhas, semente {SEMENTE}")
    print(f"%mulheres: amostra {100*(rel.sexo=='Mulher').mean():.1f}% | população {100*(pop.sexo=='Mulher').mean():.1f}%")
    print(f"idade média: amostra {rel.idade.mean():.2f} | população {pop.idade.mean():.2f}")


if __name__ == "__main__":
    main()

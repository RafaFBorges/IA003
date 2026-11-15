#!/usr/bin/env python3
"""Gera a base ENXUTA usada pelo app Streamlit  ->  app/dados/

Lê a base limpa (dados/tratados/CAGEDMOV<mes>_clean.parquet, gerada por scripts/limpeza.py)
e grava em app/dados/ (nunca toca em dados/original/):
  populacao.parquet   10 colunas em tipos compactos (~25 MB; cabe no GitHub, que recusa arquivos >100 MB)
  limpeza_log.csv     cópia do log da limpeza (o app mostra o resumo)
  limpeza_meta.json   cópia dos metadados (hash do original, parâmetros, piso salarial)

Uso:  .venv/bin/python scripts/gerar_dados_app.py [--mes 202609]
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_TRAT = RAIZ / "dados" / "tratados"
DIR_APP = RAIZ / "app" / "dados"

ORD_INSTR = ["Analfabeto", "Até 5ª Incompleto", "5ª Completo Fundamental", "6ª a 9ª Fundamental",
             "Fundamental Completo", "Médio Incompleto", "Médio Completo", "Superior Incompleto",
             "Superior Completo", "Pós-Graduação completa", "Mestrado", "Doutorado"]
ORD_REGIAO = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste", "Não identificada"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mes", default="202609")
    mes = ap.parse_args().mes

    cols = ["regiao_nome", "uf_nome", "sexo_r", "raca", "instrucao", "grande_grupo", "setor",
            "idade", "horascontratuais", "salario_mensal", "admissao"]
    d = pd.read_parquet(DIR_TRAT / f"CAGEDMOV{mes}_clean.parquet", columns=cols)

    out = pd.DataFrame({
        # rótulos pequenos como categorias (poucos valores distintos -> arquivo e memória pequenos)
        "regiao": d["regiao_nome"].astype(str).replace({"(sem rótulo)": "Não identificada"}),
        "uf": d["uf_nome"].astype(str).replace({"Minas gerais": "Minas Gerais", "Não identificado": "Não identificada"}),
        "sexo": d["sexo_r"].astype(str),
        "raca": d["raca"].astype(str),
        "instrucao": d["instrucao"].astype(str),
        "grande_grupo": d["grande_grupo"].astype(str),
        "setor": d["setor"].astype(str),
        "movimentacao": d["admissao"].map({True: "Admissão", False: "Desligamento"}),
        "idade": d["idade"].round().astype("uint8"),
        "horas": d["horascontratuais"].astype("float32"),
        "salario": d["salario_mensal"].astype("float32"),
    })
    out["regiao"] = pd.Categorical(out["regiao"], categories=ORD_REGIAO, ordered=False)
    out["instrucao"] = pd.Categorical(out["instrucao"], categories=ORD_INSTR, ordered=True)
    for c in ["uf", "sexo", "raca", "grande_grupo", "setor", "movimentacao"]:
        out[c] = out[c].astype("category")

    DIR_APP.mkdir(parents=True, exist_ok=True)
    destino = DIR_APP / "populacao.parquet"
    out.to_parquet(destino, compression="zstd", index=False)
    shutil.copy(DIR_TRAT / f"limpeza_log_{mes}.csv", DIR_APP / "limpeza_log.csv")
    shutil.copy(DIR_TRAT / f"limpeza_meta_{mes}.json", DIR_APP / "limpeza_meta.json")

    # confere que o arquivo relido bate com o que foi gerado
    rel = pd.read_parquet(destino)
    assert len(rel) == len(out) == json.loads((DIR_APP / "limpeza_meta.json").read_text())["linhas_clean"]
    print(f"{destino.relative_to(RAIZ)}: {len(rel):,} linhas, {destino.stat().st_size/1e6:.1f} MB, "
          f"{rel.memory_usage(deep=True).sum()/1e6:.0f} MB na memória")


if __name__ == "__main__":
    main()

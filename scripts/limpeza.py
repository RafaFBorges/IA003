#!/usr/bin/env python3
"""Limpeza do Novo CAGED (arquivo MOV) -> dados/tratados/

Lê SOMENTE dados/original/ (nunca altera) e grava em dados/tratados/:
  CAGEDMOV<AAAAMM>_clean.parquet   tabela limpa (+ variável `salario_mensal` e colunas derivadas)
  limpeza_log_<AAAAMM>.csv         quantas linhas cada regra excluiu/corrigiu
  limpeza_meta_<AAAAMM>.json       parâmetros, hash do arquivo original e contagens
  excluidas_R5b_R6_<AAAAMM>.csv    as linhas excluídas por R5b e R6 (auditoria)

Uso:  .venv/bin/python scripts/limpeza.py [--mes 202609]

REGRAS (decididas com o usuário; podem ser revistas, por isso ficam em constantes):
  R1  exclui unidade de pagamento Tarefa, Variável e Não identificado (sem escala mensal)
  R2  exclui horascontratuais = 0
  R3  idade ausente -> IMPUTA a mediana da mesma `categoria` de trabalhador (ex.: 103 = aprendiz); marca idade_imputada = 1
  R4  exclui salário/valor fixo <= 0
  R5  unidade Hora com "valor da hora" >= 20% do salário mensal típico da ocupação:
      o valor digitado é, na prática, um salário MENSAL -> salario_mensal = valor digitado
      (flag salario_corrigido = 1). Se o valor passa de 2x o salário mensal típico, exclui.
  R6  salario_mensal > 100x a mediana mensal da ocupação -> exclui
      (não há teto fixo: salários altos mas compatíveis com a ocupação permanecem)

`salario_mensal` é calculado aqui, a partir de valorsaláriofixo e da unidade de pagamento:
  Mês x1 | Hora x horas semanais x5 | Dia x30 | Semana x5 | Quinzena x2
e conferido contra a coluna `salário` informada pelo Ministério (resultado no log/meta).
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_ORIG = RAIZ / "dados" / "original"
DIR_TRAT = RAIZ / "dados" / "tratados"

# ---- parâmetros das regras ---------------------------------------------------------------------
UNID_EXCLUIDAS = {6, 7, 99}        # 6 Tarefa, 7 Variável, 99 Não identificado
UNID_HORA = 1
FATOR_FIXO = {2: 30.0, 3: 5.0, 4: 2.0, 5: 1.0}   # Dia, Semana, Quinzena, Mês (Hora depende das horas semanais)
SEMANAS_POR_MES = 5.0              # convenção observada nos dados do Ministério (hora e semana)
K_HORA_MENSAL = 0.20               # valor/hora >= 20% do salário mensal típico => provável salário mensal
V_REL_MAX = 2.0                    # acima de 2x o salário mensal típico nem como mensal se encaixa => exclui
K_ATIPICO = 100.0                  # salário mensal > 100x a mediana da ocupação => exclui
MIN_REF = 30                       # mínimo de linhas "Mês" para usar a mediana de uma ocupação como referência
MIN_OBS_IDADE = 30                 # mínimo de idades observadas numa categoria para usar a mediana dela na imputação
FRACAO_PISO_REF = 0.5              # a referência de uma ocupação nunca fica abaixo de 50% do piso salarial do mês (evita referências corrompidas)

# Nomes dos grandes grupos da CBO 2002 (1º dígito do código). Vêm do conhecimento da CBO, não do layout do CAGED.
GRANDE_GRUPO = {"0": "Forças armadas", "1": "Dirigentes", "2": "Profissionais das ciências e artes",
                "3": "Técnicos de nível médio", "4": "Serviços administrativos", "5": "Serviços e vendedores",
                "6": "Agropecuária", "7": "Produção industrial e construção", "8": "Produção contínua",
                "9": "Manutenção e reparação"}
ORD_INSTR = ["Analfabeto", "Até 5ª Incompleto", "5ª Completo Fundamental", "6ª a 9ª Fundamental",
             "Fundamental Completo", "Médio Incompleto", "Médio Completo", "Superior Incompleto",
             "Superior Completo", "Pós-Graduação completa", "Mestrado", "Doutorado"]


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 24), b""):
            h.update(bloco)
    return h.hexdigest()


def dicionario(layout: Path, aba: str) -> dict:
    t = pd.read_excel(layout, sheet_name=aba, dtype=str).iloc[:, :2].dropna()
    t.columns = ["cod", "desc"]
    return {str(c).strip(): d for c, d in zip(t["cod"], t["desc"])}


def referencia_mensal(mes: pd.DataFrame, cbo: pd.Series) -> pd.Series:
    """Mediana mensal da ocupação (CBO 6 dígitos; se faltar amostra, 4 e depois 2 dígitos), usando só linhas pagas por Mês."""
    ref = pd.Series(np.nan, index=cbo.index)
    for nivel in (6, 4, 2):
        g = mes.groupby(mes["cbo"].str[:nivel])["valorsaláriofixo"].agg(["median", "size"])
        g = g[g["size"] >= MIN_REF]["median"]
        novo = cbo.str[:nivel].map(g)
        ref = ref.fillna(novo)
    return ref


def main(mes_ref: str) -> None:
    arq = DIR_ORIG / f"CAGEDMOV{mes_ref}.txt"
    layout = DIR_ORIG / "Layout_Novo_Caged_Movimentacao.xlsx"
    assert arq.exists() and layout.exists(), "arquivos originais não encontrados"
    DIR_TRAT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(arq, sep=";", decimal=",", encoding="utf-8",
                     dtype={"cbo2002ocupação": str, "subclasse": str, "seção": str})
    n_raw = len(df)
    df["cbo"] = df["cbo2002ocupação"].str.zfill(6)
    log = []
    auditoria = []   # linhas excluídas por R5b/R6, para conferência manual
    cols_aud = ["cbo", "unidadesaláriocódigo", "horascontratuais", "valorsaláriofixo", "salário", "salario_mensal", "ref_mensal_ocupacao"]

    def registra(regra, acao, n_afetadas):
        log.append({"regra": regra, "acao": acao, "linhas_afetadas": int(n_afetadas),
                    "pct_do_raw": round(n_afetadas / n_raw * 100, 4), "linhas_restantes": len(df)})

    log.append({"regra": "Base original (raw)", "acao": "-", "linhas_afetadas": 0, "pct_do_raw": 0.0, "linhas_restantes": n_raw})

    # R1 ------------------------------------------------------------------------------------------
    m = df["unidadesaláriocódigo"].isin(UNID_EXCLUIDAS)
    df = df[~m]; registra("R1 unidade de pagamento Tarefa/Variável/Não identificado", "excluídas", m.sum())
    # R2 ------------------------------------------------------------------------------------------
    m = df["horascontratuais"] <= 0
    df = df[~m]; registra("R2 horas contratuais = 0", "excluídas", m.sum())
    # R3 (IMPUTAÇÃO, não exclusão) ------------------------------------------------------------------
    # Idade ausente não é aleatória (mais comum em aprendizes), então usa a mediana da MESMA `categoria` de trabalhador
    # (101 empregado geral, 103 aprendiz, 106 temporário, 111 intermitente...). Categorias com poucas idades observadas
    # (< MIN_OBS_IDADE) usam a mediana geral.
    df = df.copy()
    m = df["idade"].isna()
    mediana_geral = float(df["idade"].median())
    g = df.groupby("categoria")["idade"].agg(["median", "count"])
    medianas_idade = g["median"].where(g["count"] >= MIN_OBS_IDADE, mediana_geral)
    df["idade_imputada"] = m.astype(int)
    df.loc[m, "idade"] = df.loc[m, "categoria"].map(medianas_idade)
    n_fallback = int(df.loc[m, "idade"].isna().sum())
    df["idade"] = df["idade"].fillna(mediana_geral)                 # categoria sem mediana própria -> mediana geral
    registra("R3 idade ausente: imputada pela mediana da mesma categoria de trabalhador",
             "imputadas (" + ", ".join(f"cat {int(k)}: {v:g}" for k, v in medianas_idade.items() if k in set(df.loc[m, "categoria"])) + ")", m.sum())
    n_imputadas_por_cat = df.loc[m, "categoria"].value_counts().to_dict()
    # R4 ------------------------------------------------------------------------------------------
    m = (df["salário"] <= 0) | (df["valorsaláriofixo"] <= 0)
    df = df[~m]; registra("R4 salário ou valor fixo <= 0", "excluídas", m.sum())
    df = df.copy()

    # salario_mensal calculado a partir do valor fixo e da unidade --------------------------------
    u = df["unidadesaláriocódigo"]
    fator = u.map(FATOR_FIXO)
    df["salario_mensal"] = df["valorsaláriofixo"] * fator
    eh_hora = u == UNID_HORA
    df.loc[eh_hora, "salario_mensal"] = (df.loc[eh_hora, "valorsaláriofixo"] * df.loc[eh_hora, "horascontratuais"] * SEMANAS_POR_MES)
    df["salario_corrigido"] = 0

    # conferência com a coluna `salário` do Ministério (informativa; não exclui nada) -------------
    dif = (df["salario_mensal"] - df["salário"]).abs()
    concorda = (dif <= 0.05 + 1e-4 * df["salário"].abs())
    pct_concorda = float(concorda.mean() * 100)
    log.append({"regra": "Conferência: salario_mensal calculado x `salário` do Ministério", "acao": f"{pct_concorda:.4f}% das linhas concordam",
                "linhas_afetadas": int((~concorda).sum()), "pct_do_raw": round((~concorda).sum() / n_raw * 100, 4), "linhas_restantes": len(df)})

    # referência mensal por ocupação -------------------------------------------------------------
    ref = referencia_mensal(df[df["unidadesaláriocódigo"] == 5], df["cbo"])
    piso = float(df.loc[df["unidadesaláriocódigo"] == 5, "valorsaláriofixo"].mode().iloc[0])   # salário mais frequente = piso do mês
    ref = ref.clip(lower=FRACAO_PISO_REF * piso)
    df["ref_mensal_ocupacao"] = ref

    # R5 ------------------------------------------------------------------------------------------
    v_rel = df["valorsaláriofixo"] / ref
    cand = eh_hora & (v_rel >= K_HORA_MENSAL)
    corrige = cand & (v_rel <= V_REL_MAX)
    df.loc[corrige, "salario_mensal"] = df.loc[corrige, "valorsaláriofixo"]
    df.loc[corrige, "salario_corrigido"] = 1
    registra(f"R5 Hora com valor/hora >= {K_HORA_MENSAL:.0%} do salário mensal típico (digitado como mensal)", "corrigidas (salario_mensal = valor digitado)", corrige.sum())
    exclui = cand & ~corrige
    auditoria.append(df.loc[exclui, cols_aud].assign(regra="R5b"))
    df = df[~exclui]; registra(f"R5b Hora com valor/hora > {V_REL_MAX:g}x o salário mensal típico (não se encaixa nem como mensal)", "excluídas", exclui.sum())

    # R6 ------------------------------------------------------------------------------------------
    m = df["salario_mensal"] > K_ATIPICO * df["ref_mensal_ocupacao"]
    n_sem_ref = int(df["ref_mensal_ocupacao"].isna().sum())
    auditoria.append(df.loc[m, cols_aud].assign(regra="R6"))
    df = df[~m]; registra(f"R6 salário mensal > {K_ATIPICO:g}x a mediana da ocupação", "excluídas", m.sum())
    log.append({"regra": "Info: linhas sem ocupação de referência (R5/R6 não se aplicam)", "acao": "mantidas", "linhas_afetadas": n_sem_ref,
                "pct_do_raw": round(n_sem_ref / n_raw * 100, 4), "linhas_restantes": len(df)})

    # colunas derivadas --------------------------------------------------------------------------
    df = df.drop(columns=["ref_mensal_ocupacao"]).copy()
    d = {a: dicionario(layout, a) for a in ["sexo", "raçacor", "graudeinstrução", "uf", "seção", "tamestabjan", "unidadesaláriocódigo", "região"]}
    rot = lambda col, aba: df[col].astype(str).map(d[aba]).fillna("(sem rótulo)")
    cbo_nome = dicionario(layout, "cbo2002ocupação")
    cbo_nome = {str(k).zfill(6): v for k, v in cbo_nome.items()}

    df["unidade_pagamento"] = rot("unidadesaláriocódigo", "unidadesaláriocódigo")
    df["sexo_r"] = rot("sexo", "sexo")
    df["raca"] = rot("raçacor", "raçacor")
    df["instrucao"] = rot("graudeinstrução", "graudeinstrução")
    df["uf_nome"] = rot("uf", "uf")
    df["regiao_nome"] = rot("região", "região")
    df["setor"] = rot("seção", "seção")
    df["tam_estab"] = rot("tamestabjan", "tamestabjan")
    df["ocupacao"] = df["cbo"].map(cbo_nome).fillna("(sem nome)")
    df["grande_grupo"] = df["cbo"].str[0].map(GRANDE_GRUPO)
    df["mulher"] = df["sexo"] == 3
    df["admissao"] = df["saldomovimentação"] == 1
    df["faixa_idade"] = pd.cut(df["idade"], [13, 17, 24, 29, 39, 49, 59, 120],
                               labels=["14–17", "18–24", "25–29", "30–39", "40–49", "50–59", "60+"])
    df["escolaridade_ord"] = df["instrucao"].map({n: i + 1 for i, n in enumerate(ORD_INSTR)})
    df["log10_salario"] = np.log10(df["salario_mensal"])

    for c in ["unidade_pagamento", "sexo_r", "raca", "instrucao", "uf_nome", "regiao_nome", "setor", "tam_estab",
              "ocupacao", "grande_grupo", "seção"]:
        df[c] = df[c].astype("category")
    df = df.reset_index(drop=True)

    # verificações finais ------------------------------------------------------------------------
    assert df["salario_mensal"].gt(0).all() and df["idade"].notna().all() and df["horascontratuais"].gt(0).all()
    assert not df["unidadesaláriocódigo"].isin(UNID_EXCLUIDAS).any()

    aud = pd.concat(auditoria)
    aud["ocupacao"] = aud["cbo"].map(cbo_nome)
    aud["razao_a_mediana"] = (aud["salario_mensal"] / aud["ref_mensal_ocupacao"]).round(1)
    aud.sort_values("salario_mensal", ascending=False).to_csv(DIR_TRAT / f"excluidas_R5b_R6_{mes_ref}.csv", index=False, encoding="utf-8-sig")

    saida = DIR_TRAT / f"CAGEDMOV{mes_ref}_clean.parquet"
    df.to_parquet(saida, compression="zstd", index=False)
    pd.DataFrame(log).to_csv(DIR_TRAT / f"limpeza_log_{mes_ref}.csv", index=False, encoding="utf-8")
    meta = {"mes": mes_ref, "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "arquivo_original": arq.name, "sha256_original": sha256(arq),
            "linhas_raw": n_raw, "linhas_clean": len(df), "pct_mantido": round(len(df) / n_raw * 100, 4),
            "concordancia_salario_ministerio_pct": round(pct_concorda, 4), "piso_detectado": piso,
            "medianas_idade_por_categoria": {str(int(k)): float(v) for k, v in medianas_idade.items()},
            "idade_imputada_por_categoria": {str(int(k)): int(v) for k, v in n_imputadas_por_cat.items()},
            "mediana_geral_idade": mediana_geral,
            "parametros": {"UNID_EXCLUIDAS": sorted(UNID_EXCLUIDAS), "K_HORA_MENSAL": K_HORA_MENSAL, "V_REL_MAX": V_REL_MAX,
                           "K_ATIPICO": K_ATIPICO, "MIN_REF": MIN_REF, "MIN_OBS_IDADE": MIN_OBS_IDADE, "FRACAO_PISO_REF": FRACAO_PISO_REF, "SEMANAS_POR_MES": SEMANAS_POR_MES}}
    (DIR_TRAT / f"limpeza_meta_{mes_ref}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 95)
    print(pd.DataFrame(log).to_string(index=False))
    print(f"\nClean: {len(df):,} de {n_raw:,} linhas ({len(df)/n_raw:.2%}) -> {saida.relative_to(RAIZ)} ({saida.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mes", default="202609", help="competência AAAAMM (padrão: 202609)")
    main(ap.parse_args().mes)

"""Dimensionamento e sorteio da amostra. Fonte única: usado por scripts/sorteio_amostra.py e pelo app.

Regras:
  n0 = z^2 * p * (1 - p) / e^2                       (proporção, população infinita)
  n  = ceil(n0)                                      se a amostra for até 5% da população (regra usual: a correção é desprezível)
  n  = ceil(n0 / (1 + (n0 - 1) / N))                 se passar de 5% (correção para população finita)
  sorteio = numpy.random.default_rng(semente).choice(N, n, replace=False)   (AAS sem reposição)

A amostra é UMA só e fixa (semente fixa): o script grava o CSV e o app apenas o lê.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

DIR_DADOS = Path(__file__).resolve().parent / "dados"
LIMITE_CORRECAO = 0.05   # só corrige para população finita se n0 passar de 5% de N


def tamanho_amostra(confianca: float, erro: float, p: float, N: int) -> dict:
    z = float(stats.norm.ppf(1 - (1 - confianca) / 2))
    n0 = z**2 * p * (1 - p) / erro**2
    usa_correcao = n0 / N > LIMITE_CORRECAO
    n = math.ceil(n0 / (1 + (n0 - 1) / N)) if usa_correcao else math.ceil(n0)
    return dict(z=z, n0=n0, n=n, usa_correcao=usa_correcao, n_com_correcao=math.ceil(n0 / (1 + (n0 - 1) / N)))


def indices_aas(N: int, n: int, semente: int) -> np.ndarray:
    """Posições (ordenadas) das n movimentações sorteadas, sem reposição, com a semente dada."""
    return np.sort(np.random.default_rng(semente).choice(N, size=n, replace=False))


def padronizar(amostra: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Dá à amostra lida do CSV os mesmos tipos da população (categorias e ordem da escolaridade)."""
    am = amostra.copy()
    for c in pop.columns:
        if c not in am.columns:
            continue
        if isinstance(pop[c].dtype, pd.CategoricalDtype):
            am[c] = pd.Categorical(am[c], categories=pop[c].cat.categories, ordered=pop[c].cat.ordered)
        else:
            am[c] = am[c].astype(pop[c].dtype)
    return am


@st.cache_data
def carregar_meta_amostra() -> dict:
    return json.loads((DIR_DADOS / "amostra_meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def amostra_oficial(_pop: pd.DataFrame) -> pd.DataFrame:
    am = pd.read_csv(DIR_DADOS / "amostra_aas.csv")
    return padronizar(am.drop(columns=["id_populacao"]), _pop).assign(id_populacao=am["id_populacao"].to_numpy())

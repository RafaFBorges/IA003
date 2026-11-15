"""Carga de dados, formatação e estilo dos gráficos (compartilhado pelas abas)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DIR_DADOS = Path(__file__).resolve().parent / "dados"

# Fonte e identificação da base (citadas na aba População e na entrega)
FONTE_URL = "ftp://ftp.mtps.gov.br/pdet/microdados/NOVO%20CAGED/2026/202609/CAGEDMOV202609.7z"
FONTE_PASTA = "ftp://ftp.mtps.gov.br/pdet/microdados/NOVO%20CAGED"
MES_ROTULO = "setembro de 2026"
REPO_URL = "https://github.com/RafaFBorges/IA003"

# Cor única (matiz azul) para magnitudes; o cinza serve de apoio (grade, referência)
COR = "#2a78d6"
COR_DESTAQUE = "#eb6834"   # reservada para marcar uma referência (ex.: média) sobre o azul
COR_POP = "#8c8b87"        # cinza neutro: a população como referência nas comparações com a amostra (a amostra é o azul)


@st.cache_resource(show_spinner="Carregando a população (4,25 milhões de movimentações)…")
def carregar_populacao() -> pd.DataFrame:
    """Lê a base enxuta uma única vez por processo. Não alterar o DataFrame retornado."""
    return pd.read_parquet(DIR_DADOS / "populacao.parquet")


@st.cache_data
def carregar_meta() -> dict:
    return json.loads((DIR_DADOS / "limpeza_meta.json").read_text(encoding="utf-8"))


@st.cache_data
def carregar_log() -> pd.DataFrame:
    return pd.read_csv(DIR_DADOS / "limpeza_log.csv")


def n_br(x: float, casas: int = 0) -> str:
    """Número no padrão brasileiro: 1.234.567,89"""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def brl(x: float, casas: int = 0) -> str:
    return "R$ " + n_br(x, casas)


def pct(x: float, casas: int = 1) -> str:
    return n_br(100 * x, casas) + "%"


def estilo_figura(fig, altura=380, legenda=False):
    """Aplica o visual comum: fundo transparente (segue o tema claro/escuro), grade discreta."""
    fig.update_layout(
        height=altura, margin=dict(l=8, r=8, t=36, b=8), showlegend=legenda,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), title=dict(font=dict(size=15), x=0, xanchor="left"),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.22)", zeroline=False)
    return fig

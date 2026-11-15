"""App Streamlit do trabalho IA003: população, amostra e inferência sobre o Novo CAGED (setembro/2026).

Rodar:  .venv/bin/streamlit run app/app.py
"""
import streamlit as st

import aba_populacao
from comum import MES_ROTULO, carregar_populacao

st.set_page_config(page_title="IA003 · Novo CAGED", page_icon="📊", layout="wide")

st.title("Movimentações do emprego formal no Brasil")
st.caption(f"Novo CAGED, Ministério do Trabalho e Emprego · competência {MES_ROTULO} · cada linha é uma admissão ou um desligamento")

df = carregar_populacao()

tab_pop, tab_amo, tab_inf = st.tabs(["População", "Amostra", "Inferência"])

with tab_pop:
    aba_populacao.render(df)

with tab_amo:
    st.header("Amostra")
    st.info("Em construção: aqui entram o cálculo do tamanho da amostra, o sorteio (AAS) e a análise descritiva "
            "da amostra, comparada com a população.")

with tab_inf:
    st.header("Inferência")
    st.info("Em construção: aqui entram as hipóteses (proporções e médias), intervalos de confiança, testes, "
            "condições, comparação com os parâmetros da população e limitações.")

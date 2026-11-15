"""App Streamlit do trabalho IA003: população, amostra e inferência sobre o Novo CAGED (setembro/2026).

Rodar:  .venv/bin/streamlit run app/app.py
"""
import streamlit as st

import aba_amostra
import aba_extras
import aba_inferencia
import aba_populacao
from comum import MES_ROTULO, carregar_populacao

st.set_page_config(page_title="IA003 · Novo CAGED", page_icon="📊", layout="wide")

st.title("Movimentações do emprego formal no Brasil")
st.caption(f"Novo CAGED, Ministério do Trabalho e Emprego · competência {MES_ROTULO} · cada linha é uma admissão ou um desligamento")

df = carregar_populacao()

tab_pop, tab_amo, tab_inf, tab_ext = st.tabs(["População", "Amostra", "Inferência", "Informações extras"])

with tab_pop:
    aba_populacao.render(df)

with tab_amo:
    aba_amostra.render(df)

with tab_inf:
    aba_inferencia.render(df)

with tab_ext:
    aba_extras.render(df)

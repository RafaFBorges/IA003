"""Aba 2 - Amostra: dimensionamento e análise descritiva da amostra (AAS fixa) comparada com a população (Painel 2)."""
import textwrap
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aba_populacao import QUALITATIVAS, QUANTITATIVAS, medidas_resumo, parametros_populacao, tabela_frequencias
from amostragem import amostra_oficial, carregar_meta_amostra, tamanho_amostra
from comum import COR, COR_POP, brl, estilo_figura, n_br, pct

POP_GRAFICO = "População"


# ----------------------------------------------------------------------------------------------
# cálculos da população (uma vez só; não dependem da amostra)
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def freq_populacao(_df, col: str, ordenada: bool) -> pd.DataFrame:
    return tabela_frequencias(_df, col, ordenada)


@st.cache_data(show_spinner=False)
def medidas_populacao(_df, var: str) -> dict:
    x = _df[var].to_numpy(dtype="float64")
    m = medidas_resumo(x)
    if var == "salario":
        m["geometrica"] = float(10 ** np.log10(x).mean())
    return m


def bins_variavel(df: pd.DataFrame, var: str):
    """Faixas do histograma (iguais para população e amostra) + rótulos e ticks do eixo."""
    if var == "idade":
        bordas = np.arange(13.5, df["idade"].max() + 2.5, 2.0)
        rot = [f"{a + .5:.0f} a {b - .5:.0f} anos" for a, b in zip(bordas[:-1], bordas[1:])]
        return bordas, rot, None, None, "Idade (anos, faixas de 2)"
    if var == "horas":
        bordas = np.arange(0, 62, 2.0)
        rot = [f"{a:.0f} a {b:.0f} h" for a, b in zip(bordas[:-1], bordas[1:])]
        return bordas, rot, None, None, "Horas semanais contratadas (faixas de 2 h)"
    lx = np.log10(df["salario"].to_numpy(dtype="float64"))
    bordas = np.arange(np.floor(lx.min() * 10) / 10, lx.max() + 0.1, 0.1)
    rot = [f"R$ {n_br(10 ** a)} a R$ {n_br(10 ** b)}" for a, b in zip(bordas[:-1], bordas[1:])]
    vals = np.array([500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 500000])
    vals = vals[(np.log10(vals) >= bordas[0]) & (np.log10(vals) <= bordas[-1])]
    return bordas, rot, np.log10(vals), [n_br(v) for v in vals], "Salário mensal (R$, escala logarítmica)"


def contar(x: np.ndarray, bordas: np.ndarray) -> np.ndarray:
    return np.histogram(x, bins=bordas)[0]


@st.cache_data(show_spinner=False)
def hist_populacao(_df, var: str) -> np.ndarray:
    bordas = bins_variavel(_df, var)[0]
    x = _df[var].to_numpy(dtype="float64")
    return contar(np.log10(x) if var == "salario" else x, bordas)


# ----------------------------------------------------------------------------------------------
# gráficos
# ----------------------------------------------------------------------------------------------
def fig_barras_comparadas(t: pd.DataFrame, titulo: str) -> go.Figure:
    """t: categoria, pct_am, pct_pop, n_am. Barras agrupadas: amostra (azul) e população (cinza)."""
    rot = ["<br>".join(textwrap.wrap(c, 38)) for c in t["categoria"]]
    custom = np.stack([t["categoria"], t["n_am"].map(n_br)], axis=-1)
    poucas = len(t) <= 12
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=t["pct_am"] * 100, y=rot, orientation="h", name="Amostra", marker_color=COR, customdata=custom,
        text=[pct(p) for p in t["pct_am"]] if poucas else None, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Amostra: %{x:.2f}% (%{customdata[1]} movimentações)<extra></extra>"))
    fig.add_trace(go.Bar(
        x=t["pct_pop"] * 100, y=rot, orientation="h", name=POP_GRAFICO, marker_color=COR_POP, customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b><br>População: %{x:.2f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(visible=False, range=[0, max(t["pct_am"].max(), t["pct_pop"].max()) * 100 * 1.2])
    fig.update_layout(title=titulo, barmode="group", bargap=0.3, bargroupgap=0.08,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return estilo_figura(fig, altura=max(300, 58 * len(t) + 80), legenda=True)


def fig_hist_comparado(centros, pct_am, cont_am, pct_pop, largura, rotulos, xtitulo, titulo,
                       tickvals=None, ticktext=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centros, y=pct_am, width=largura, name="Amostra", marker_color=COR, marker_line_width=0,
        customdata=np.stack([rotulos, cont_am], axis=-1),
        hovertemplate="%{customdata[0]}<br>Amostra: %{y:.2f}% (%{customdata[1]:,.0f} movimentações)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=centros, y=pct_pop, mode="lines", name=POP_GRAFICO, line=dict(color=COR_POP, width=2.5),
        customdata=np.array(rotulos), hovertemplate="%{customdata}<br>População: %{y:.2f}%<extra></extra>"))
    fig.update_xaxes(title_text=xtitulo)
    if tickvals is not None:
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
    fig.update_yaxes(title_text="% das movimentações")
    fig.update_layout(title=titulo, bargap=0.05, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return estilo_figura(fig, altura=380, legenda=True)


# ----------------------------------------------------------------------------------------------
# seções
# ----------------------------------------------------------------------------------------------
CSS_CARTOES = """
<style>
.cartoes{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:4px 0 16px}
.cartao{position:relative;border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:14px 10px 16px;
        text-align:center;cursor:help;outline:none}
.cartao:hover,.cartao:focus{border-color:#2a78d6}
.cartao .t{font-size:.9rem;opacity:.75}
.cartao .v{font-size:2rem;font-weight:600;line-height:1.25;margin-top:4px}
.cartao .dica{visibility:hidden;opacity:0;position:absolute;left:50%;top:calc(100% + 8px);transform:translateX(-50%);
              width:max(280px,100%);background:#262730;color:#fff;font-size:.85rem;line-height:1.45;
              padding:10px 12px;border-radius:8px;text-align:left;z-index:1000;transition:opacity .12s;
              box-shadow:0 4px 14px rgba(0,0,0,.3);pointer-events:none}
.cartao:hover .dica,.cartao:focus .dica{visibility:visible;opacity:1}
</style>
"""


def cartoes_parametros(itens):
    """Cartões lado a lado: título, valor centralizado e, ao passar o mouse, o texto explicativo (itens: titulo, valor, dica)."""
    html = "".join(
        f'<div class="cartao" tabindex="0"><div class="t">{escape(t)} ⓘ</div><div class="v">{escape(v)}</div>'
        f'<div class="dica">{escape(d)}</div></div>' for t, v, d in itens)
    st.html(CSS_CARTOES + f'<div class="cartoes">{html}</div>')


def secao_dimensionamento(N: int, meta: dict, n_arquivo: int):
    st.subheader("Tamanho da amostra")
    t = tamanho_amostra(meta["confianca"], meta["margem_erro"], meta["p"], N)
    j = meta["justificativas"]
    cartoes_parametros([
        ("População (N)", n_br(N),
         f"{n_br(N)} movimentações no total (a população limpa). A amostra é {pct(t['n'] / N, 3)} dela. "
         "Como é menos de 5%, a correção para população finita é desprezível e não é aplicada."
         if not t["usa_correcao"] else
         f"{n_br(N)} movimentações no total (a população limpa). A amostra passa de 5% dela, então aplicamos a correção para população finita."),
        ("Tamanho da amostra (n)", n_br(t["n"]),
         f"Para estimar uma proporção com a margem de erro desejada: n = z² · p · (1 − p) / e² = "
         f"{n_br(t['z'], 2)}² × {n_br(meta['p'], 1)} × {n_br(1 - meta['p'], 1)} / {n_br(meta['margem_erro'], 2)}² = "
         f"{n_br(t['n0'], 2)}, arredondado para cima: n = {n_br(t['n'])}. Confira na calculadora de tamanho de amostra da USP/Bauru com os mesmos parâmetros."),
        ("Confiança", pct(meta["confianca"], 0), f"z = {n_br(t['z'], 2)}. {j['confianca']}"),
        ("Margem de erro", pct(meta["margem_erro"], 0), j["margem_erro"]),
        ("Proporção esperada (p)", n_br(meta["p"], 1), j["p"]),
    ])
    st.caption("Fórmula, sorteio e semente estão na aba **Informações extras**.")
    if n_arquivo != t["n"]:
        st.warning(f"O arquivo da amostra tem {n_br(n_arquivo)} linhas, mas o cálculo dá {n_br(t['n'])}. "
                   "Rode scripts/sorteio_amostra.py para atualizar.")


def secao_indicadores(am: pd.DataFrame, par: dict):
    st.subheader("Indicadores da amostra e da população")
    c = st.columns(5)
    pm, pa = float((am["sexo"] == "Mulher").mean()), float((am["movimentacao"] == "Admissão").mean())
    im, sm = float(am["idade"].mean()), float(am["salario"].mean())
    geo = float(10 ** np.log10(am["salario"].to_numpy(dtype="float64")).mean())

    def dpp(a, b):
        return f"{(a - b) * 100:+.2f} p.p. vs população".replace(".", ",")

    c[0].metric("% mulheres", pct(pm, 2), delta=dpp(pm, par["p_mulher"]), delta_color="off", help=f"População: {pct(par['p_mulher'], 2)}")
    c[1].metric("% admissões", pct(pa, 2), delta=dpp(pa, par["p_adm"]), delta_color="off", help=f"População: {pct(par['p_adm'], 2)}")
    c[2].metric("Idade média", f"{n_br(im, 2)} anos", delta=f"{im - par['idade_media']:+.2f} vs população".replace(".", ","),
                delta_color="off", help=f"População: {n_br(par['idade_media'], 2)} anos")
    c[3].metric("Salário médio", brl(sm, 2), delta=f"{brl(sm - par['sal_media'], 2)} vs população".replace("R$ -", "-R$ "),
                delta_color="off", help=f"População: {brl(par['sal_media'], 2)}")
    c[4].metric("Salário médio geométrico", brl(geo, 2), delta=f"{brl(geo - par['sal_geo'], 2)} vs população".replace("R$ -", "-R$ "),
                delta_color="off", help=f"População: {brl(par['sal_geo'], 2)}")
    st.caption(f"Amostra de {n_br(len(am))} movimentações. A diferença para a população é o **erro amostral**: "
               "não é erro de medição, vem de olhar só uma parte.")


def secao_qualitativas(pop: pd.DataFrame, am: pd.DataFrame):
    st.subheader("Variáveis qualitativas: amostra × população")
    escolha = st.segmented_control("Variável", list(QUALITATIVAS), default="Sexo", key="amo_qual",
                                   label_visibility="collapsed") or "Sexo"
    col, ordenada = QUALITATIVAS[escolha]
    tp = freq_populacao(pop, col, ordenada).rename(columns={"n": "n_pop", "pct": "pct_pop"})
    vc = am[col].value_counts(sort=False).rename_axis("categoria").reset_index(name="n_am")
    vc["categoria"] = vc["categoria"].astype(str)
    t = tp.merge(vc, on="categoria", how="left").fillna({"n_am": 0})
    t["n_am"] = t["n_am"].astype(int)
    t["pct_am"] = t["n_am"] / len(am)
    t["dif"] = (t["pct_am"] - t["pct_pop"]) * 100
    g, tab = st.columns([3, 2])
    with g:
        st.plotly_chart(fig_barras_comparadas(t, f"{escolha}: % das movimentações"), width="stretch", config={"displayModeBar": False})
    with tab:
        tt = pd.DataFrame({escolha: t["categoria"], "n (amostra)": t["n_am"], "% amostra": t["pct_am"] * 100,
                           "% população": t["pct_pop"] * 100, "Diferença (p.p.)": t["dif"]})
        st.dataframe(tt, hide_index=True, width="stretch", height=min(40 + 35 * len(tt), 600),
                     column_config={"% amostra": st.column_config.NumberColumn(format="%.2f%%"),
                                    "% população": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Diferença (p.p.)": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Categorias raras podem sumir da amostra ou aparecer a mais por acaso. Com poucas observações, a % da amostra oscila bastante.")


def secao_quantitativas(pop: pd.DataFrame, am: pd.DataFrame):
    st.subheader("Variáveis quantitativas: amostra × população")
    escolha = st.segmented_control("Variável quantitativa", QUANTITATIVAS, default="Idade", key="amo_quant",
                                   label_visibility="collapsed") or "Idade"
    var = {"Idade": "idade", "Salário mensal": "salario", "Horas contratuais": "horas"}[escolha]
    unid = {"idade": lambda v: f"{n_br(v, 2)} anos", "horas": lambda v: f"{n_br(v, 2)} h",
            "salario": lambda v: brl(v, 2)}[var]
    bordas, rot, tickvals, ticktext, xtitulo = bins_variavel(pop, var)
    xa = am[var].to_numpy(dtype="float64")
    ca = contar(np.log10(xa) if var == "salario" else xa, bordas)
    cp = hist_populacao(pop, var)
    centros = (bordas[:-1] + bordas[1:]) / 2
    largura = (bordas[1] - bordas[0]) * 0.95
    ma = medidas_resumo(xa)
    mp = medidas_populacao(pop, var)
    if var == "salario":
        ma["geometrica"] = float(10 ** np.log10(xa).mean())
    g, tab = st.columns([3, 2])
    with g:
        if var == "horas":
            st.warning(f"**{pct(float(np.mean(np.isclose(pop['horas'].to_numpy(), 44))), 1)} das movimentações da população têm 44 horas.** "
                       "O pico único é esperado nesta variável.")
        st.plotly_chart(fig_hist_comparado(centros, ca / len(xa) * 100, ca, cp / len(pop) * 100, largura, rot, xtitulo,
                                           f"{escolha}: amostra (barras) e população (linha)", tickvals, ticktext),
                        width="stretch", config={"displayModeBar": False})
        if var == "salario":
            st.caption("Escala logarítmica. A cauda de salários muito altos tem poucas observações, então a amostra "
                       "quase não a mostra: é nela que a média amostral varia mais de sorteio para sorteio.")
    with tab:
        linhas = [("n", n_br(ma["n"]), n_br(mp["n"]), ""),
                  ("Média", unid(ma["media"]), unid(mp["media"]), unid(ma["media"] - mp["media"])),
                  ("Mediana", unid(ma["mediana"]), unid(mp["mediana"]), unid(ma["mediana"] - mp["mediana"]))]
        if var == "salario":
            linhas.append(("Média geométrica", unid(ma["geometrica"]), unid(mp["geometrica"]), unid(ma["geometrica"] - mp["geometrica"])))
        linhas += [("Desvio-padrão", unid(ma["dp"]), unid(mp["dp"]), unid(ma["dp"] - mp["dp"])),
                   ("Mínimo", unid(ma["minimo"]), unid(mp["minimo"]), ""),
                   ("1º quartil (Q1)", unid(ma["q1"]), unid(mp["q1"]), unid(ma["q1"] - mp["q1"])),
                   ("3º quartil (Q3)", unid(ma["q3"]), unid(mp["q3"]), unid(ma["q3"] - mp["q3"])),
                   ("Máximo", unid(ma["maximo"]), unid(mp["maximo"]), ""),
                   ("Assimetria", n_br(ma["assimetria"], 2), n_br(mp["assimetria"], 2), n_br(ma["assimetria"] - mp["assimetria"], 2))]
        st.dataframe(pd.DataFrame(linhas, columns=["Medida", "Amostra", "População", "Diferença"]), hide_index=True,
                     width="stretch", height=40 + 35 * len(linhas))


def render(pop: pd.DataFrame):
    meta = carregar_meta_amostra()
    par = parametros_populacao(pop)
    am = amostra_oficial(pop)
    st.header("Amostra")
    secao_dimensionamento(len(pop), meta, len(am))
    st.divider()
    secao_indicadores(am, par)
    secao_qualitativas(pop, am)
    secao_quantitativas(pop, am)

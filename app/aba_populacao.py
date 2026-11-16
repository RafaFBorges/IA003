"""Aba 1 - População: análise descritiva da base inteira (Painel 1 do trabalho)."""
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from comum import COR, COR_DESTAQUE, brl, carregar_meta, estilo_figura, n_br, pct

# variável qualitativa -> (coluna, rótulo, ordem natural?)
QUALITATIVAS = {
    "Sexo": ("sexo", False),
    "Raça/cor": ("raca", False),
    "Escolaridade": ("instrucao", True),
    "Admissão × desligamento": ("movimentacao", False),
    "Grande grupo ocupacional": ("grande_grupo", False),
    "Setor (CNAE)": ("setor", False),
    "Região": ("regiao", False),
    "UF": ("uf", False),
}
QUANTITATIVAS = ["Idade", "Salário mensal", "Horas contratuais"]

DICIONARIO = pd.DataFrame([
    ("Sexo", "Qualitativa nominal", "Homem ou Mulher, como declarado pelo empregador."),
    ("Raça/cor", "Qualitativa nominal", "Branca, Preta, Parda, Amarela, Indígena ou Não informada."),
    ("Escolaridade", "Qualitativa ordinal", "Grau de instrução em 12 níveis, de Analfabeto a Doutorado."),
    ("Admissão × desligamento", "Qualitativa nominal", "Tipo da movimentação: entrada (admissão) ou saída (desligamento)."),
    ("Grande grupo ocupacional", "Qualitativa nominal", "1º dígito da CBO 2002 (10 grandes grupos, de Dirigentes a Manutenção)."),
    ("Setor (CNAE)", "Qualitativa nominal", "Seção da CNAE 2.0 da atividade do estabelecimento (22 seções)."),
    ("Região / UF", "Qualitativa nominal", "5 regiões e 27 UFs onde fica o estabelecimento."),
    ("Idade", "Quantitativa discreta", "Idade em anos. Ausente em 0,12% das linhas, preenchida pela mediana da mesma categoria de trabalhador."),
    ("Salário mensal", "Quantitativa contínua", "Variável criada na limpeza: salário convertido para valor mensal, em R$ (hora × horas semanais × 5; dia × 30; semana × 5; quinzena × 2; mês × 1)."),
    ("Horas contratuais", "Quantitativa contínua", "Horas semanais contratadas."),
], columns=["Variável", "Tipo", "O que é"])


# ----------------------------------------------------------------------------------------------
# cálculos
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def mapa_uf_regiao(_df) -> dict:
    return _df.drop_duplicates("uf").set_index("uf")["regiao"].astype(str).to_dict()


def medidas_resumo(x: np.ndarray) -> dict:
    x = x.astype("float64")
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return dict(n=len(x), media=x.mean(), dp=x.std(ddof=1), minimo=x.min(), q1=q1, mediana=med, q3=q3,
                maximo=x.max(), assimetria=float(stats.skew(x)))


@st.cache_data(show_spinner=False)
def parametros_populacao(_df) -> dict:
    """Valores da população INTEIRA (não dependem dos filtros). Servirão de parâmetros na aba Inferência."""
    sal = _df["salario"].to_numpy(dtype="float64")
    idade = _df["idade"].to_numpy(dtype="float64")
    return dict(
        n=len(_df),
        p_mulher=float((_df["sexo"] == "Mulher").mean()),
        p_adm=float((_df["movimentacao"] == "Admissão").mean()),
        idade_media=idade.mean(), idade_dp=idade.std(ddof=1),
        sal_media=sal.mean(), sal_mediana=float(np.median(sal)), sal_dp=sal.std(ddof=1),
        sal_geo=float(10 ** np.log10(sal).mean()),
        idade_mediana=float(np.median(idade)),
    )


def tabela_frequencias(d: pd.DataFrame, col: str, ordenada: bool) -> pd.DataFrame:
    s = d[col]
    vc = s.value_counts(sort=False)
    vc = vc.reindex(s.cat.categories) if ordenada else vc[vc > 0].sort_values(ascending=False)
    t = vc.rename_axis("categoria").reset_index(name="n")
    t["categoria"] = t["categoria"].astype(str)
    t["pct"] = t["n"] / t["n"].sum()
    return t


def _box_stats(v: np.ndarray) -> dict:
    """Estatísticas do boxplot de Tukey (bastões = menor/maior valor dentro de Q1-1,5·IIQ e Q3+1,5·IIQ)."""
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    iiq = q3 - q1
    inf, sup = q1 - 1.5 * iiq, q3 + 1.5 * iiq
    dentro = v[(v >= inf) & (v <= sup)]
    return dict(q1=q1, med=med, q3=q3, lf=dentro.min(), uf=dentro.max(), media=v.mean(),
                n_atip=int(len(v) - len(dentro)), p_atip=(len(v) - len(dentro)) / len(v))


# ----------------------------------------------------------------------------------------------
# gráficos
# ----------------------------------------------------------------------------------------------
def fig_barras(t: pd.DataFrame, titulo: str) -> go.Figure:
    rot = ["<br>".join(textwrap.wrap(c, 38)) for c in t["categoria"]]
    fig = go.Figure(go.Bar(
        x=t["pct"] * 100, y=rot, orientation="h", marker_color=COR,
        text=[pct(p) for p in t["pct"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([t["categoria"], t["n"].map(n_br)], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{x:.2f}% das movimentações<br>%{customdata[1]} movimentações<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(visible=False, range=[0, t["pct"].max() * 100 * 1.18])
    fig.update_layout(title=titulo, bargap=0.35)
    return estilo_figura(fig, altura=max(260, 34 * len(t) + 70))


def fig_histograma(centros, pct_bins, contagens, largura, rotulos, linhas, box, xtitulo,
                   tickvals=None, ticktext=None, titulo="", xrange=None, com_box=True) -> go.Figure:
    """Histograma (% das movimentações por faixa) com mediana/média marcadas e, opcionalmente, boxplot embaixo."""
    if com_box:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.03)
    else:
        fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(
        x=centros, y=pct_bins, width=largura, marker_color=COR, marker_line_width=0, name="Frequência",
        showlegend=False, customdata=np.stack([rotulos, contagens], axis=-1),
        hovertemplate="%{customdata[0]}<br>%{y:.2f}% das movimentações<br>%{customdata[1]:,.0f} movimentações<extra></extra>",
    ), row=1, col=1)
    ymax = float(np.max(pct_bins)) * 1.05
    for valor, nome, dash in linhas:
        fig.add_trace(go.Scatter(x=[valor, valor], y=[0, ymax], mode="lines", name=nome,
                                 line=dict(color=COR_DESTAQUE, width=2, dash=dash),
                                 hovertemplate=nome + "<extra></extra>"), row=1, col=1)
    if com_box:
        b = box
        fig.add_trace(go.Box(
            q1=[b["q1"]], median=[b["med"]], q3=[b["q3"]], lowerfence=[b["lf"]], upperfence=[b["uf"]],
            mean=[b["media"]], y=["Boxplot"], orientation="h", name="Boxplot", showlegend=False,
            marker_color=COR, line=dict(color=COR, width=1.6), fillcolor="rgba(42,120,214,0.30)",
            hoverinfo="skip",
        ), row=2, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="% das movimentações", row=1, col=1)
    xa = dict(title_text=xtitulo, row=2 if com_box else 1, col=1)
    if tickvals is not None:
        xa.update(tickvals=tickvals, ticktext=ticktext)
    fig.update_xaxes(**xa)
    if xrange:
        fig.update_xaxes(range=xrange)
    fig.update_layout(title=titulo, bargap=0.02, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return estilo_figura(fig, altura=430 if com_box else 340, legenda=True)


def _linhas_ref(media, mediana, fmt, tr=lambda v: v):
    return [(tr(mediana), f"Mediana: {fmt(mediana)}", "solid"), (tr(media), f"Média: {fmt(media)}", "dash")]


# ----------------------------------------------------------------------------------------------
# seções da aba
# ----------------------------------------------------------------------------------------------
def secao_parametros(par: dict, piso: float):
    st.subheader("Parâmetros da população")
    with st.container(border=True):
        st.caption("Valores calculados sobre a população inteira. **Não mudam com os filtros** abaixo. "
                   "São os parâmetros que as hipóteses da aba Inferência vão comparar com a amostra.")
        c = st.columns(5)
        c[0].metric("Proporção de mulheres", pct(par["p_mulher"], 2))
        c[1].metric("Proporção de admissões", pct(par["p_adm"], 2))
        c[2].metric("Idade média", f"{n_br(par['idade_media'], 2)} anos", help=f"Desvio-padrão: {n_br(par['idade_dp'], 2)} anos")
        c[3].metric("Salário médio", brl(par["sal_media"], 2), help=f"Desvio-padrão: {brl(par['sal_dp'], 2)}")
        c[4].metric("Salário médio geométrico", brl(par["sal_geo"], 2),
                    help="Média dos logaritmos do salário, voltada para R$. É o parâmetro natural de um teste feito sobre o log do salário.")
        st.caption(f"Salário mediano: {brl(par['sal_mediana'], 2)}. Piso salarial detectado nos dados: {brl(piso)}.".replace("$", "\\$"))


def filtros(df: pd.DataFrame):
    """Filtros em uma linha, acima dos gráficos. Retorna (DataFrame filtrado, descrição do filtro)."""
    st.subheader("Explorar a população")
    st.caption("Os filtros valem para todos os indicadores e gráficos desta aba. Sem seleção, vale a população inteira.")
    ufs_regiao = mapa_uf_regiao(df)
    ordem_regiao = [r for r in df["regiao"].cat.categories if r in set(ufs_regiao.values())]
    c = st.columns(5)
    regioes = c[0].multiselect("Região", ordem_regiao, key="pop_regiao", placeholder="Todas")
    opcoes_uf = sorted(u for u, r in ufs_regiao.items() if not regioes or r in regioes)
    ufs = c[1].multiselect("UF", opcoes_uf, key="pop_uf", placeholder="Todas")
    ufs = [u for u in ufs if u in opcoes_uf]
    sexos = c[2].multiselect("Sexo", sorted(df["sexo"].cat.categories), key="pop_sexo", placeholder="Todos")
    grupos = c[3].multiselect("Grande grupo", sorted(df["grande_grupo"].cat.categories), key="pop_grupo", placeholder="Todos")
    movs = c[4].multiselect("Admissão × desligamento", sorted(df["movimentacao"].cat.categories), key="pop_mov", placeholder="Todos")

    mask = None
    for col, sel in [("regiao", regioes), ("uf", ufs), ("sexo", sexos), ("grande_grupo", grupos), ("movimentacao", movs)]:
        if sel:
            m = df[col].isin(sel).to_numpy()
            mask = m if mask is None else (mask & m)
    return (df if mask is None else df[mask]), mask is not None


def secao_indicadores(d: pd.DataFrame, par: dict, filtrado: bool):
    n = len(d)
    st.subheader("Indicadores")
    if n == 0:
        return
    c = st.columns(5)
    c[0].metric("Movimentações", n_br(n), delta=f"{pct(n / par['n'], 1)} da população" if filtrado else None, delta_color="off")
    pm, pa = float((d["sexo"] == "Mulher").mean()), float((d["movimentacao"] == "Admissão").mean())
    im, sm = float(d["idade"].mean()), float(d["salario"].median())

    def dpp(a, b):  # diferença em pontos percentuais frente à população
        return f"{(a - b) * 100:+.1f} p.p. vs população".replace(".", ",") if filtrado else None

    c[1].metric("% mulheres", pct(pm), delta=dpp(pm, par["p_mulher"]), delta_color="off")
    c[2].metric("% admissões", pct(pa), delta=dpp(pa, par["p_adm"]), delta_color="off")
    c[3].metric("Idade média", f"{n_br(im, 1)} anos",
                delta=f"{im - par['idade_media']:+.1f} vs população".replace(".", ",") if filtrado else None, delta_color="off")
    c[4].metric("Salário mediano", brl(sm),
                delta=f"{brl(sm - par['sal_mediana'])} vs população".replace("R$ -", "-R$ ") if filtrado else None, delta_color="off")


def secao_qualitativas(d: pd.DataFrame):
    st.subheader("Variáveis qualitativas")
    escolha = st.segmented_control("Variável", list(QUALITATIVAS), default="Sexo", key="pop_qual", label_visibility="collapsed")
    escolha = escolha or "Sexo"
    col, ordenada = QUALITATIVAS[escolha]
    t = tabela_frequencias(d, col, ordenada)
    g, tab = st.columns([3, 2])
    with g:
        st.plotly_chart(fig_barras(t, f"{escolha}: % das {n_br(len(d))} movimentações"), width="stretch",
                        config={"displayModeBar": False})
    with tab:
        tt = t.rename(columns={"categoria": escolha, "n": "Movimentações", "pct": "%"}).copy()
        tt["%"] = (tt["%"] * 100).round(2)
        st.dataframe(tt, hide_index=True, width="stretch", height=min(40 + 35 * len(tt), 600),
                     column_config={"Movimentações": st.column_config.NumberColumn(format="localized"),
                                    "%": st.column_config.NumberColumn(format="%.2f%%")})
        if col == "raca":
            st.caption("'Não informada' é uma categoria do próprio cadastro: a raça/cor não foi declarada.")


def _tabela_medidas(m: dict, unid, extra_geo=None):
    linhas = [("Movimentações (n)", n_br(m["n"])), ("Média", unid(m["media"])), ("Mediana", unid(m["mediana"])),
              ("Desvio-padrão", unid(m["dp"])), ("Mínimo", unid(m["minimo"])), ("1º quartil (Q1)", unid(m["q1"])),
              ("3º quartil (Q3)", unid(m["q3"])), ("Máximo", unid(m["maximo"])),
              ("Assimetria", n_br(m["assimetria"], 2))]
    if extra_geo is not None:
        linhas.insert(3, ("Média geométrica", unid(extra_geo)))
    st.dataframe(pd.DataFrame(linhas, columns=["Medida", "Valor"]), hide_index=True, width="stretch", height=40 + 35 * len(linhas))


def secao_quantitativas(d: pd.DataFrame):
    st.subheader("Variáveis quantitativas")
    escolha = st.segmented_control("Variável quantitativa", QUANTITATIVAS, default="Idade", key="pop_quant",
                                   label_visibility="collapsed") or "Idade"
    g, tab = st.columns([3, 2])

    if escolha == "Idade":
        x = d["idade"].to_numpy()
        m = medidas_resumo(x)
        cont = np.bincount(x, minlength=int(x.max()) + 1)
        idades = np.arange(len(cont))
        sel = cont > 0
        box = _box_stats(x.astype("float64"))
        fig = fig_histograma(idades[sel], cont[sel] / len(x) * 100, cont[sel], 0.9,
                             [f"{i} anos" for i in idades[sel]], _linhas_ref(m["media"], m["mediana"], lambda v: f"{n_br(v, 1)} anos"),
                             box, "Idade (anos)", titulo="Distribuição da idade")
        with g:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"No boxplot, os pontos além dos bastões seriam valores atípicos: {n_br(box['n_atip'])} movimentações "
                       f"({pct(box['p_atip'])}). Idade imputada (0,12% das linhas) segue a mediana da categoria do trabalhador.")
        with tab:
            _tabela_medidas(m, lambda v: f"{n_br(v, 2)} anos")

    elif escolha == "Horas contratuais":
        x = d["horas"].to_numpy()
        m = medidas_resumo(x)
        bordas = np.arange(0, 62, 2.0)
        cont, _ = np.histogram(x, bins=bordas)
        centros = (bordas[:-1] + bordas[1:]) / 2
        box = _box_stats(x.astype("float64"))
        p44 = float(np.mean(np.isclose(x, 44)))
        fig = fig_histograma(centros, cont / len(x) * 100, cont, 1.9,
                             [f"{a:.0f} a {b:.0f} h" for a, b in zip(bordas[:-1], bordas[1:])],
                             _linhas_ref(m["media"], m["mediana"], lambda v: f"{n_br(v, 1)} h"),
                             box, "Horas semanais contratadas (faixas de 2 h)", titulo="Distribuição das horas contratuais")
        with g:
            st.warning(f"**{pct(p44, 1)} das movimentações têm exatamente 44 horas semanais** (a jornada padrão). "
                       "Por isso o histograma tem um pico único e o boxplot quase não tem caixa: "
                       "Q1, mediana e Q3 coincidem em 44. Esta variável varia pouco.")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"Fora dos bastões do boxplot: {n_br(box['n_atip'])} movimentações ({pct(box['p_atip'])}), "
                       "quase todas com jornada menor que 44 horas (tempo parcial, 30, 36 horas).")
        with tab:
            _tabela_medidas(m, lambda v: f"{n_br(v, 2)} h")

    else:  # Salário mensal
        x = d["salario"].to_numpy(dtype="float64")
        m = medidas_resumo(x)
        lx = np.log10(x)
        geo = float(10 ** lx.mean())
        lo = np.floor(lx.min() * 10) / 10
        bordas = np.arange(lo, lx.max() + 0.05, 0.05)
        cont, _ = np.histogram(lx, bins=bordas)
        centros = (bordas[:-1] + bordas[1:]) / 2
        vals = np.array([500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 500000])
        vals = vals[(np.log10(vals) >= lo) & (np.log10(vals) <= lx.max())]
        box = _box_stats(lx)
        linhas = _linhas_ref(geo, m["mediana"], lambda v: brl(v, 0), tr=np.log10)
        linhas[1] = (np.log10(geo), f"Média geométrica: {brl(geo)}", "dash")
        fig = fig_histograma(centros, cont / len(x) * 100, cont, 0.048,
                             [f"R$ {n_br(10 ** a)} a R$ {n_br(10 ** b)}" for a, b in zip(bordas[:-1], bordas[1:])],
                             linhas, box, "Salário mensal (R$, escala logarítmica)",
                             tickvals=np.log10(vals), ticktext=[n_br(v) for v in vals],
                             titulo="Distribuição do salário mensal, em escala logarítmica")
        with g:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Escala logarítmica: cada passo no eixo multiplica o salário (de R$ 1.000 para R$ 10.000 é a mesma distância "
                       "de R$ 10.000 para R$ 100.000). Ela mostra o corpo da distribuição e a cauda de salários altos no mesmo gráfico. "
                       f"O boxplot é do logaritmo do salário; {n_br(box['n_atip'])} movimentações ({pct(box['p_atip'])}) ficam além dos bastões.")
            lim = st.slider("Zoom em escala linear: salários até (R$)", 2000, 20000, 6000, 500, key="pop_zoom")
            bl = np.linspace(0, lim, 81)
            cl, _ = np.histogram(x, bins=bl)
            cc = (bl[:-1] + bl[1:]) / 2
            ref = [(v, n, d_) for v, n, d_ in _linhas_ref(m["media"], m["mediana"], lambda v: brl(v)) if v <= lim]
            fig2 = fig_histograma(cc, cl / len(x) * 100, cl, lim / 80 * 0.95,
                                  [f"R$ {n_br(a)} a R$ {n_br(b)}" for a, b in zip(bl[:-1], bl[1:])], ref, None,
                                  "Salário mensal (R$, escala linear)", titulo=f"Zoom: salários de até {brl(lim)}", com_box=False)
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
            st.caption(f"O zoom mostra {pct(float(np.mean(x <= lim)))} das movimentações; a altura das barras continua sendo "
                       "o % do total. A média fica acima da mediana porque poucos salários muito altos puxam a média para cima.")
        with tab:
            _tabela_medidas(m, lambda v: brl(v, 2), extra_geo=geo)
            st.caption("A assimetria positiva e alta confirma a cauda longa à direita. Por isso a média e o desvio-padrão "
                       "do salário são sensíveis a poucos valores; a mediana e a média geométrica são mais estáveis.")


def render(df: pd.DataFrame):
    meta = carregar_meta()
    par = parametros_populacao(df)
    st.header("População")
    st.caption("Como a população foi definida, a limpeza e o dicionário de variáveis estão na aba **Informações extras**.")
    secao_parametros(par, meta["piso_detectado"])
    st.divider()
    d, filtrado = filtros(df)
    if len(d) == 0:
        st.warning("Nenhuma movimentação atende a essa combinação de filtros. Retire algum filtro.")
        return
    secao_indicadores(d, par, filtrado)
    secao_qualitativas(d)
    secao_quantitativas(d)

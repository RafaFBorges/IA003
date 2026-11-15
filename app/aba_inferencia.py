"""Aba Inferência: cinco testes de hipóteses com a amostra oficial, comparados com a população."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import inferencia as inf
from amostragem import amostra_oficial
from comum import COR, COR_DESTAQUE, COR_POP, brl, estilo_figura, n_br, pct

MENOS = "−"


# ----------------------------------------------------------------------------------------------
# formatação
# ----------------------------------------------------------------------------------------------
def md(txt: str) -> str:
    """Escapa o cifrão para o Markdown não tratá-lo como fórmula."""
    return txt.replace("$", "\\$")


def fp(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else n_br(p, 4)


def p_txt(p: float) -> str:
    return f"p {fp(p)}" if p < 1e-4 else f"p = {fp(p)}"


def sinal(x: float, casas: int, unid: str) -> str:
    s = n_br(abs(x), casas)
    if float(s.replace(".", "").replace(",", ".")) == 0:
        return s + unid
    return ("+" if x > 0 else MENOS) + s + unid


def fmt_dif(h: dict, d: float) -> str:
    """Diferença A − B na unidade mostrada: pontos percentuais ou % da média geométrica."""
    x = float(inf.exibir(h, d))
    return sinal(x, 2, " pp") if h["tipo"] == "prop" else sinal(x, 1, "%")


def fmt_ic(h: dict, ic) -> str:
    return f"[{fmt_dif(h, ic[0])}; {fmt_dif(h, ic[1])}]"


def fmt_grupo(h: dict, x: float) -> str:
    """Valor de um grupo: proporção ou média geométrica do salário."""
    return pct(x, 2) if h["tipo"] == "prop" else brl(10**x, 2)


def estat_txt(h: dict, r: dict) -> str:
    if h["tipo"] == "prop":
        return f"z = {n_br(r['stat'], 2)}"
    return f"t = {n_br(r['stat'], 2)} (gl = {n_br(r['gl'], 0)})"


def avaliacao(rej: bool, verdadeira: bool):
    """(rótulo curto, frase) comparando a decisão da amostra com o que vale na população."""
    if rej and verdadeira:
        return "Acerto", "a decisão de rejeitar H0 está correta: a diferença existe na população, no sentido de H1."
    if rej and not verdadeira:
        return "Falso positivo", ("rejeitar H0 foi uma conclusão errada: na população a diferença não está no sentido de H1. "
                                  "É o efeito da variação amostral.")
    if not rej and verdadeira:
        return "Não detectou (erro tipo II)", ("a amostra **não detectou** uma diferença que existe na população. "
                                               "O teste tem pouco poder para uma diferença desse tamanho com n = 2.401.")
    return "Acerto (H1 falsa)", "não rejeitar H0 é coerente com a população: a diferença não está no sentido de H1."


# ----------------------------------------------------------------------------------------------
# população e simulação (calculadas uma vez por processo)
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Calculando os parâmetros da população e simulando 1.000 amostras…")
def populacao_e_simulacao(_pop: pd.DataFrame, n: int) -> dict:
    v = inf.vetores(_pop)
    verdade = {h["id"]: inf.diferenca_populacao(h, v) for h in inf.HIPOTESES}
    return dict(verdade=verdade, sim=inf.simular(v, n))


# ----------------------------------------------------------------------------------------------
# gráficos
# ----------------------------------------------------------------------------------------------
TICKS_R = [0.01, 1, 10, 100, 1000, 10000, 100000]


def fig_qq(y: np.ndarray, titulo: str, em_log: bool = False, ytitulo: str = None, altura: int = 320) -> go.Figure:
    """QQ plot contra a normal. Reta de referência pelos quartis (robusta a caudas)."""
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    z1, z3 = stats.norm.ppf([0.25, 0.75])
    q1, q3 = np.percentile(y, [25, 75])
    incl = (q3 - q1) / (z3 - z1)
    xs = np.array([q[0], q[-1]])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=y, mode="markers", name="Dados", marker=dict(color=COR, size=4, opacity=0.55),
                             hovertemplate="quantil teórico %{x:.2f}<br>valor %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=q1 + incl * (xs - z1), mode="lines", name="Reta da normal (pelos quartis)",
                             line=dict(color=COR_POP, width=2.5), hoverinfo="skip"))
    fig.update_xaxes(title_text="Quantil teórico da normal")
    if em_log:
        fig.update_yaxes(title_text="Salário em R$ (escala log)", tickvals=[np.log10(t) for t in TICKS_R],
                         ticktext=[n_br(t, 2 if t < 1 else 0) for t in TICKS_R], range=[y.min() - 0.15, y.max() + 0.15])
    else:
        fig.update_yaxes(title_text=ytitulo or "Salário (R$)")
    fig.update_layout(title=titulo, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return estilo_figura(fig, altura=altura, legenda=True)


def fig_floresta(itens, titulo: str, xtitulo: str) -> go.Figure:
    """itens: (hipótese, resultado da amostra, parâmetro da população). Ponto = amostra com IC; losango cinza = população."""
    ys = [h["curto"] for h, _, _ in itens]
    est = np.array([inf.exibir(h, r["dif"]) for h, r, _ in itens])
    lo = np.array([inf.exibir(h, r["ic"][0]) for h, r, _ in itens])
    hi = np.array([inf.exibir(h, r["ic"][1]) for h, r, _ in itens])
    pop = np.array([inf.exibir(h, p["dif"]) for h, _, p in itens])
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color="rgba(128,128,128,.7)", width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=est, y=ys, mode="markers", name="Amostra (IC de 95%)", marker=dict(color=COR, size=10),
        error_x=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo, color=COR, thickness=2, width=7),
        customdata=np.stack([lo, hi], axis=-1),
        hovertemplate="<b>%{y}</b><br>Amostra: %{x:.2f}<br>IC: %{customdata[0]:.2f} a %{customdata[1]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=pop, y=ys, mode="markers", name="População (valor verdadeiro)",
        marker=dict(color=COR_POP, size=11, symbol="diamond"),
        hovertemplate="<b>%{y}</b><br>População: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title_text=xtitulo, zeroline=False, showgrid=True, gridcolor="rgba(128,128,128,0.22)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title=titulo, legend=dict(orientation="h", y=-0.35, yanchor="top", x=0, xanchor="left"))
    return estilo_figura(fig, altura=120 + 62 * len(itens), legenda=True)


def fig_variabilidade(h: dict, sim: dict, r_amostra: dict) -> go.Figure:
    dif = inf.exibir(h, sim["res"][h["id"]]["dif"])
    verd = float(inf.exibir(h, sim["verdade"][h["id"]]))
    nossa = float(inf.exibir(h, r_amostra["dif"]))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dif, nbinsx=40, name=f"{n_br(sim['B'])} amostras (diferença estimada)", marker_color=COR,
                               marker_line_width=0, hovertemplate="diferença %{x:.2f}: %{y} amostras<extra></extra>"))
    ymax = float(np.histogram(dif, bins=40)[0].max()) * 1.12
    for x, nome, cor, tr in [(0, "H0: sem diferença", "rgba(128,128,128,.8)", "dot"),
                             (verd, "População (valor verdadeiro)", COR_POP, "dash"),
                             (nossa, "Nossa amostra", COR_DESTAQUE, "solid")]:
        fig.add_trace(go.Scatter(x=[x, x], y=[0, ymax], mode="lines", name=nome, line=dict(color=cor, width=2.5, dash=tr),
                                 hovertemplate=f"{nome}: %{{x:.2f}}<extra></extra>"))
    fig.update_xaxes(title_text="Diferença A − B (pp)" if h["tipo"] == "prop" else "Diferença A − B (% da média geométrica)")
    fig.update_yaxes(title_text="Número de amostras")
    fig.update_layout(title=f"{h['titulo']}: como a diferença varia de amostra para amostra", bargap=0.03,
                      legend=dict(orientation="h", y=-0.28, yanchor="top", x=0, xanchor="left"))
    return estilo_figura(fig, altura=420, legenda=True)


# ----------------------------------------------------------------------------------------------
# cada hipótese
# ----------------------------------------------------------------------------------------------
def tabela_condicoes(h: dict, r: dict, N: int) -> pd.DataFrame:
    n = r["n1"] + r["n2"]
    linhas = [
        ("Amostra aleatória simples", "Sorteio sem reposição com semente fixa (ver aba Informações extras).", "Atende"),
        ("Grupos independentes",
         f"Cada movimentação está em um só grupo e a amostra é {pct(n / N, 3)} da população (menos de 10%).", "Atende"),
    ]
    if h["tipo"] == "prop":
        menor = min(r["n1"] * r["pc"], r["n1"] * (1 - r["pc"]), r["n2"] * r["pc"], r["n2"] * (1 - r["pc"]))
        linhas.append(("Sucessos e fracassos esperados ≥ 10 em cada grupo",
                       f"n = {n_br(r['n1'])} e {n_br(r['n2'])}; proporção combinada sob H0 = {pct(r['pc'], 2)}; "
                       f"menor valor esperado = {n_br(menor, 0)}.", "Atende" if menor >= 10 else "Não atende"))
    else:
        linhas.append(("Tamanho de cada grupo (teorema central do limite)",
                       f"n = {n_br(r['n1'])} e {n_br(r['n2'])} (mínimo recomendado: 30).",
                       "Atende" if min(r["n1"], r["n2"]) >= 30 else "Não atende"))
        ok = abs(r["assim1"]) < 1 and abs(r["assim2"]) < 1
        linhas.append(("Distribuição do log10 do salário sem assimetria forte",
                       f"Assimetria: {n_br(r['assim1'], 2)} ({h['a']}) e {n_br(r['assim2'], 2)} ({h['b']}). "
                       + ("" if ok else "Há uma cauda esquerda pesada: poucos salários muito baixos (veja o QQ plot). "
                                        "Com n grande, a média continua aproximadamente normal e o teste não assume variâncias iguais (Welch)."),
                       "Atende" if ok else "Atenção"))
    return pd.DataFrame(linhas, columns=["Condição", "Como foi verificada", "Resultado"])


def robustez(h: dict, ya: np.ndarray, yb: np.ndarray, r: dict):
    """Repete o teste de médias sem os salários abaixo de R$ 100 (log10 < 2)."""
    ya2, yb2 = ya[ya >= 2], yb[yb >= 2]
    removidas = len(ya) - len(ya2) + len(yb) - len(yb2)
    r2 = inf.welch(ya2, yb2, h["alt"])
    mesma = (r2["p"] < inf.ALFA) == (r["p"] < inf.ALFA)
    st.caption(md(
        f"**Verificação de robustez:** sem as {n_br(removidas)} movimentações com salário abaixo de R$ 100, a diferença é "
        f"{fmt_dif(h, r2['dif'])} ({p_txt(r2['p'])}) e a decisão {'é a mesma' if mesma else '**muda**'}."))
    return mesma


def secao_hipotese(h: dict, v_am: dict, N: int, pop_h: dict, sim: dict):
    ya, yb = inf.grupos(h, v_am)
    r = inf.executar(h, ya, yb)
    rej = r["p"] < inf.ALFA
    verd = inf.h1_verdadeira(h, pop_h["dif"])
    cobre = r["ic"][0] <= pop_h["dif"] <= r["ic"][1]
    rotulo_av, frase_av = avaliacao(rej, verd)
    res_sim = sim["res"][h["id"]]
    prop = h["tipo"] == "prop"
    sa, sb = h["sa"], h["sb"]
    gt = ">" if h["alt"] == "greater" else "<"

    st.markdown(f"#### {h['titulo']}")
    st.markdown(f"*{md(h['pergunta'])}*")

    # 1. hipóteses
    if prop:
        h0 = f"p({sa}) = p({sb})"
        h1 = f"p({sa}) {gt} p({sb})"
        legenda = (f"p é a proporção de movimentações que são {h['eventos']} em cada grupo. "
                   f"Teste z de duas proporções, unilateral, α = {pct(inf.ALFA, 0)}.")
    else:
        h0 = f"μ({sa}) = μ({sb})"
        h1 = f"μ({sa}) {gt} μ({sb})"
        legenda = (f"μ é a média do log10 do salário mensal em cada grupo (diferença nula = médias geométricas iguais). "
                   f"Teste t de Welch, unilateral, α = {pct(inf.ALFA, 0)}.")
    c1, c2 = st.columns(2)
    c1.markdown(f"**H0:** {h0}")
    c2.markdown(f"**H1:** {h1}")
    st.caption(md(f"Grupo A = {h['a']}; grupo B = {h['b']}. {legenda}"))

    # 2. condições
    resultado_cond = tabela_condicoes(h, r, N)["Resultado"]
    if (resultado_cond == "Não atende").any():
        status = "**há condição não atendida**"
    elif (resultado_cond == "Atenção").any():
        status = "atendidas, com uma **atenção** (assimetria do log10 do salário, vista no QQ plot abaixo)"
    else:
        status = "todas atendidas"
    st.markdown("**Condições e normalidade**")
    st.caption(f"Condições do teste: {status}. A tabela com a verificação de cada uma está na aba **Informações extras**.")
    sr = sim["res"][h["id"]]
    with st.expander("QQ plots", expanded=True):
        q = st.columns(3)
        if prop:
            q[0].plotly_chart(fig_qq(100 * sr["est1"], f"Proporção: {h['a']}", ytitulo="Proporção na amostra (%)"),
                              width="stretch", key=f"qq_{h['id']}_a")
            q[1].plotly_chart(fig_qq(100 * sr["est2"], f"Proporção: {h['b']}", ytitulo="Proporção na amostra (%)"),
                              width="stretch", key=f"qq_{h['id']}_b")
            leg = ("Como temos a população inteira, sorteamos 1.000 amostras do mesmo tamanho e guardamos a proporção de cada grupo "
                   "e a diferença. Pontos sobre a reta indicam distribuição amostral aproximadamente normal, o que sustenta o teste z.")
        else:
            q[0].plotly_chart(fig_qq(ya, f"log10 do salário: {h['a']}", True), width="stretch", key=f"qq_{h['id']}_a")
            q[1].plotly_chart(fig_qq(yb, f"log10 do salário: {h['b']}", True), width="stretch", key=f"qq_{h['id']}_b")
            leg = ("Os dois primeiros gráficos mostram os **dados** (salários na amostra); o miolo segue razoavelmente a reta e os desvios "
                   "ficam nas pontas, sobretudo na esquerda, onde há poucos salários muito baixos (a escala logarítmica os destaca). "
                   "Os pequenos degraus vêm de salários repetidos, como o piso. O terceiro mostra a **distribuição amostral** da diferença "
                   "em 1.000 amostras: sendo aproximadamente normal, o teste t é adequado apesar da cauda esquerda dos dados.")
        q[2].plotly_chart(fig_qq(inf.exibir(h, sr["dif"]), "Diferença A − B (1.000 amostras)",
                                 ytitulo="Diferença (pp)" if prop else "Diferença (% da média geométrica)"),
                          width="stretch", key=f"qq_{h['id']}_dif")
        st.caption(md(leg))

    # 3. resultado
    st.markdown("**Resultado do teste**")
    m = st.columns(5)
    m[0].metric("Diferença na amostra (A − B)", fmt_dif(h, r["dif"]),
                help=("Pontos percentuais (pp) entre as duas taxas." if prop else
                      f"Diferença relativa entre as médias geométricas ({fmt_grupo(h, r['est1'])} contra {fmt_grupo(h, r['est2'])}). "
                      f"Em log10: {n_br(r['dif'], 4)}."))
    m[1].metric("IC de 95% da diferença", fmt_ic(h, r["ic"]), help="Intervalo bilateral, calculado sem supor H0.")
    m[2].metric("Estatística de teste", estat_txt(h, r))
    m[3].metric("Valor-p (unilateral)", fp(r["p"]), help="Probabilidade, sob H0, de uma diferença ao menos tão extrema quanto a observada, no sentido de H1.")
    m[4].metric("Decisão (α = 5%)", "Rejeita H0" if rej else "Não rejeita H0")
    robusta = None if prop else robustez(h, ya, yb, r)

    # 4. comparação com a população
    st.markdown("**Comparação com a população**")
    tab = pd.DataFrame({
        "": [h["a"], h["b"], "Diferença (A − B)"],
        "Amostra": [fmt_grupo(h, r["est1"]), fmt_grupo(h, r["est2"]), fmt_dif(h, r["dif"])],
        "População": [fmt_grupo(h, pop_h["est1"]), fmt_grupo(h, pop_h["est2"]), fmt_dif(h, pop_h["dif"])],
    })
    st.dataframe(tab, hide_index=True, width="stretch")
    if not prop:
        st.caption("Salários em média geométrica (valor típico, menos sensível aos extremos que a média aritmética).")

    # 5. conclusão
    st.markdown("**Conclusão**")
    if rej:
        concl = f"Com α = 5%, **rejeitamos H0**: há evidência estatística de que {h['h1_frase']}."
    else:
        concl = (f"Com α = 5%, **não rejeitamos H0**: não há evidência estatística suficiente de que {h['h1_frase']}. "
                 "Isso não prova que os grupos são iguais; mostra apenas que a diferença vista na amostra pode ser fruto do acaso.")
    st.markdown(md(
        f"{concl} O IC de 95% da diferença é {fmt_ic(h, r['ic'])}"
        f"{' e inclui o zero' if r['ic'][0] <= 0 <= r['ic'][1] else ' e não inclui o zero'}.\n\n"
        f"Na população inteira a diferença é {fmt_dif(h, pop_h['dif'])}. O IC {'contém' if cobre else '**não** contém'} esse valor, "
        f"e a H1 é {'verdadeira' if verd else 'falsa'} na população: {frase_av} "
        f"Em {n_br(sim['B'])} amostras do mesmo tamanho, {pct(res_sim['rejeita'].mean(), 1)} rejeitariam H0."))
    st.caption(md(f"Limitações deste teste. {h['limites']}"))
    return dict(h=h, r=r, pop=pop_h, rej=rej, verdadeira=verd, cobre=cobre, rotulo=rotulo_av, robusta=robusta)


# ----------------------------------------------------------------------------------------------
# seções gerais
# ----------------------------------------------------------------------------------------------
def secao_introducao(am: pd.DataFrame, N: int):
    st.caption(md(
        f"Amostra oficial: n = {n_br(len(am))} de {n_br(N)} movimentações · α = {pct(inf.ALFA, 0)} · IC de {pct(inf.CONFIANCA, 0)} · "
        "valor-p unilateral. A justificativa dos métodos e o QQ plot do salário estão na aba **Informações extras**."))


def secao_resumo(resultados: list):
    st.subheader("Resumo e comparação com a população")
    linhas = []
    for x in resultados:
        h, r, p = x["h"], x["r"], x["pop"]
        linhas.append({
            "Hipótese": f"{h['id']}. {h['titulo']}",
            "H1": h["h1_frase"][0].upper() + h["h1_frase"][1:],
            "Diferença na amostra": fmt_dif(h, r["dif"]),
            "IC de 95%": fmt_ic(h, r["ic"]),
            "Valor-p": fp(r["p"]),
            "Decisão (α = 5%)": "Rejeita H0" if x["rej"] else "Não rejeita H0",
            "Diferença na população": fmt_dif(h, p["dif"]),
            "H1 verdadeira na população?": "Sim" if x["verdadeira"] else "Não",
            "IC contém o valor da população?": "Sim" if x["cobre"] else "Não",
            "Avaliação": x["rotulo"],
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    props = [(x["h"], x["r"], x["pop"]) for x in resultados if x["h"]["tipo"] == "prop"]
    meds = [(x["h"], x["r"], x["pop"]) for x in resultados if x["h"]["tipo"] == "media"]
    c1.plotly_chart(fig_floresta(props, "Taxas (diferença A − B)", "Pontos percentuais"), width="stretch", key="flor_prop")
    c2.plotly_chart(fig_floresta(meds, "Salários (diferença A − B)", "% da média geométrica"), width="stretch", key="flor_med")
    st.markdown(md(
        "**Como ler.** O ponto azul é a diferença observada na amostra e a barra é o IC de 95%. O losango cinza é o valor verdadeiro na "
        "população, que só conhecemos porque temos a base inteira (numa pesquisa real, ele é desconhecido). Quando a barra passa pelo zero, "
        "não há evidência de diferença; quando o losango está dentro da barra, o IC acertou o parâmetro."))
    certas = sum(x["rotulo"].startswith("Acerto") for x in resultados)
    st.markdown(md(
        f"**Balanço:** {certas} das {len(resultados)} decisões coincidem com o que vale na população. "
        "Nas demais, a amostra, que na prática seria tudo o que se tem, não detectou uma diferença real (erro tipo II) ou apontou uma que não existe (falso positivo)."))


def secao_variabilidade(sim: dict, resultados: list):
    st.subheader("Variabilidade amostral")
    st.markdown(md(
        f"Nossa conclusão vem de **uma única amostra**. Para ver quanto ela poderia ter variado, sorteamos {n_br(sim['B'])} amostras "
        f"aleatórias simples, cada uma com n = {n_br(sim['n'])}, e aplicamos os mesmos cinco testes em todas "
        f"(semente da simulação: {sim['semente']}, diferente da semente da amostra oficial)."))
    linhas = []
    for x in resultados:
        h = x["h"]
        rs = sim["res"][h["id"]]
        d = inf.exibir(h, rs["dif"])
        lo, hi = np.percentile(d, [2.5, 97.5])
        linhas.append({
            "Hipótese": f"{h['id']}. {h['curto']}",
            "Diferença na população": fmt_dif(h, x["pop"]["dif"]),
            "H1 verdadeira na população?": "Sim" if x["verdadeira"] else "Não",
            "Amostras que rejeitam H0": pct(rs["rejeita"].mean(), 1),
            "Variação da diferença (95% das amostras)": f"{sinal(lo, 2, '')} a {sinal(hi, 2, '')}",
            "ICs que contêm o valor da população": pct(rs["cobre"].mean(), 1),
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    st.markdown(md(
        "- Quando a H1 é **verdadeira**, a coluna \"amostras que rejeitam H0\" é o **poder** do teste. Poder baixo explica por que uma diferença real pode passar "
        "despercebida numa amostra.\n"
        "- Quando a H1 é **falsa**, essa coluna mostra com que frequência uma amostra levaria a uma conclusão errada.\n"
        "- Cerca de 95% dos intervalos de confiança devem conter o valor verdadeiro: é isso que significa \"95% de confiança\"."))
    ids = [x["h"]["id"] for x in resultados]
    nomes = {x["h"]["id"]: f"{x['h']['id']}. {x['h']['curto']}" for x in resultados}
    escolha = st.segmented_control("Hipótese", ids, default=ids[0], format_func=lambda i: nomes[i], key="inf_var")
    x = next(r for r in resultados if r["h"]["id"] == (escolha or ids[0]))
    st.plotly_chart(fig_variabilidade(x["h"], sim, x["r"]), width="stretch", key="fig_var")


def secao_limitacoes(n: int, N: int, resultados: list):
    st.subheader("Limitações")
    st.markdown(md(
        f"- **Uma amostra, um mês.** Os dados são de setembro de 2026; sazonalidade e conjuntura do mês podem não valer para outros meses. "
        f"A amostra de {n_br(n)} movimentações é só {pct(n / N, 3)} da população.\n"
        "- **Movimentações, não pessoas.** Não há identificador de pessoa: a mesma pessoa pode aparecer mais de uma vez, o que fere um pouco a independência.\n"
        "- **Só emprego formal declarado no prazo.** Trabalho informal, servidores estatutários e declarações fora do prazo ficam de fora.\n"
        "- **Significância não é importância.** Com n grande, até diferenças minúsculas podem ficar \"significativas\"; olhe o tamanho da diferença e o IC, não só o valor-p.\n"
        "- **Falta de significância não é prova de igualdade.** Quando o teste não rejeita H0, a diferença pode existir e ser pequena demais para a amostra (erro tipo II).\n"
        "- **Cinco testes na mesma amostra.** Se todas as H0 fossem verdadeiras, a chance de pelo menos um falso positivo seria cerca de 23% (1 − 0,95⁵).\n"
        "- **Comparações brutas.** Nenhum teste controla ocupação, setor, escolaridade ou região; as diferenças não indicam causa.\n"
        "- **Salários muito baixos.** A limpeza tratou os valores absurdamente altos, mas restam salários mensais irrealistas (como R$ 0,01). "
        + ("Refizemos os testes de média sem as movimentações abaixo de R$ 100 e a decisão não mudou."
           if all(x["robusta"] for x in resultados if x["robusta"] is not None) else
           "Ao refazer os testes de média sem as movimentações abaixo de R$ 100, pelo menos uma decisão mudou: trate esse resultado com cautela.")))


def render(pop: pd.DataFrame):
    am = amostra_oficial(pop)
    N = len(pop)
    pdados = populacao_e_simulacao(pop, len(am))
    st.header("Inferência")
    secao_introducao(am, N)
    st.divider()
    v_am = inf.vetores(am)
    abas = st.tabs([f"{h['id']}. {h['curto']}" for h in inf.HIPOTESES] + ["Resumo"])
    resultados = []
    for aba, h in zip(abas, inf.HIPOTESES):
        with aba:
            resultados.append(secao_hipotese(h, v_am, N, pdados["verdade"][h["id"]], pdados["sim"]))
    with abas[-1]:
        secao_resumo(resultados)
        st.divider()
        secao_variabilidade(pdados["sim"], resultados)
        st.divider()
        secao_limitacoes(len(am), N, resultados)

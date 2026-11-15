"""Testes de hipóteses para duas amostras independentes e simulação de amostras repetidas.

Sem dependência do Streamlit (pode ser testado e usado por scripts).

Cada hipótese compara o grupo A com o grupo B (diferença = A − B) e tem H1 unilateral:
  - tipo "prop":  proporção de um evento (admissão ou desligamento) em cada grupo; teste z de duas proporções
                  (erro-padrão combinado sob H0); IC da diferença com erro-padrão não combinado.
  - tipo "media": média do log10 do salário em cada grupo; teste t de Welch (não assume variâncias iguais).
O IC é bilateral de 95%; o valor-p é unilateral, na direção de H1.
"""
import numpy as np
from scipy import stats

CONFIANCA = 0.95            # a mesma confiança usada no cálculo do tamanho da amostra
ALFA = round(1 - CONFIANCA, 10)   # significância = 1 − confiança
N_SIMULACOES = 1000
SEMENTE_SIMULACAO = 20261115   # só para a simulação de variabilidade; diferente da semente da amostra


# ----------------------------------------------------------------------------------------------
# especificação das hipóteses
# ----------------------------------------------------------------------------------------------
def vetores(df) -> dict:
    """Vetores (um valor por movimentação) usados para montar os grupos de todas as hipóteses."""
    adm = (df["movimentacao"] == "Admissão").to_numpy()
    return dict(
        adm=adm,
        homem=(df["sexo"] == "Homem").to_numpy(),
        mulher=(df["sexo"] == "Mulher").to_numpy(),
        sp=(df["uf"] == "São Paulo").to_numpy(),
        velho=(df["idade"] > 40).to_numpy(),
        logsal=np.log10(df["salario"].to_numpy(dtype=np.float64)),
    )


HIPOTESES = [
    dict(
        id=1, tipo="prop", alt="greater", curto="Admissão: homens × mulheres",
        titulo="Taxa de admissão: homens × mulheres",
        pergunta="Entre as movimentações de homens, a proporção de admissões é maior do que entre as de mulheres?",
        a="Homens", b="Mulheres", sa="H", sb="M", evento="admissão", eventos="admissões",
        h1_frase="a taxa de admissão dos homens é maior que a das mulheres",
        sel=lambda v: (v["homem"], v["mulher"], v["adm"]),
        limites="Compara **taxas** de admissão, não o número de admissões (que depende do tamanho de cada grupo). "
                "Não controla ocupação nem setor.",
    ),
    dict(
        id=2, tipo="prop", alt="greater", curto="Admissão: São Paulo × resto",
        titulo="Taxa de admissão: São Paulo × demais estados",
        pergunta="Em São Paulo, a proporção de movimentações que são admissões é maior do que nos demais estados?",
        a="São Paulo", b="Demais estados", sa="SP", sb="resto", evento="admissão", eventos="admissões",
        h1_frase="a taxa de admissão em São Paulo é maior que a dos demais estados",
        sel=lambda v: (v["sp"], ~v["sp"], v["adm"]),
        limites="O estado foi escolhido **antes** de ver os resultados (o maior em movimentações). "
                "O grupo \"demais estados\" mistura 26 unidades muito diferentes entre si.",
    ),
    dict(
        id=3, tipo="prop", alt="greater", curto="Desligamento: mais de 40 anos",
        titulo="Taxa de desligamento: mais de 40 anos × 40 anos ou menos",
        pergunta="Entre as pessoas com mais de 40 anos, a proporção de movimentações que são desligamentos é maior "
                 "do que entre as de 40 anos ou menos?",
        a="Mais de 40 anos", b="40 anos ou menos", sa="&gt;40", sb="≤40", evento="desligamento", eventos="desligamentos",
        h1_frase="a taxa de desligamento de quem tem mais de 40 anos é maior que a de quem tem 40 anos ou menos",
        sel=lambda v: (v["velho"], ~v["velho"], ~v["adm"]),
        limites="O corte em 40 anos é arbitrário e dividir a idade em dois grupos perde informação. Parte das idades "
                "ausentes foi imputada na limpeza. Como desligamento = 100% − admissão, este teste não traz informação "
                "independente sobre admissões.",
    ),
    dict(
        id=4, tipo="media", alt="less", curto="Salário: admitidos × demitidos",
        titulo="Salário: admitidos × desligados",
        pergunta="O salário típico (média do log10, isto é, média geométrica) de quem foi admitido é menor do que o de quem foi desligado?",
        a="Admitidos", b="Desligados", sa="adm", sb="deslig",
        h1_frase="o salário médio geométrico dos admitidos é menor que o dos desligados",
        sel=lambda v: (v["adm"], ~v["adm"], v["logsal"]),
        limites="As linhas são movimentações, não pessoas: a mesma pessoa pode aparecer como admitida e como desligada "
                "no mês, o que enfraquece a independência entre os grupos. Admitidos e desligados têm perfis "
                "de ocupação diferentes e isso não é controlado.",
    ),
    dict(
        id=5, tipo="media", alt="less", curto="Salário: mulheres × homens",
        titulo="Salário: mulheres × homens",
        pergunta="O salário típico (média do log10, isto é, média geométrica) das mulheres é menor do que o dos homens?",
        a="Mulheres", b="Homens", sa="M", sb="H",
        h1_frase="o salário médio geométrico das mulheres é menor que o dos homens",
        sel=lambda v: (v["mulher"], v["homem"], v["logsal"]),
        limites="É uma diferença **bruta**: não controla ocupação, setor, escolaridade nem horas. Ela descreve a "
                "diferença nas movimentações do mês e **não prova discriminação**.",
    ),
]


def grupos(h: dict, v: dict, idx=None):
    """(valores do grupo A, valores do grupo B): booleanos (proporção) ou log10 do salário (média)."""
    a, b, y = h["sel"](v)
    if idx is not None:
        a, b, y = a[idx], b[idx], y[idx]
    return y[a], y[b]


# ----------------------------------------------------------------------------------------------
# testes
# ----------------------------------------------------------------------------------------------
def duas_proporcoes(x1: int, n1: int, x2: int, n2: int, alt: str, conf: float = CONFIANCA) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    dif = p1 - p2
    pc = (x1 + x2) / (n1 + n2)
    z = dif / np.sqrt(pc * (1 - pc) * (1 / n1 + 1 / n2))
    p = stats.norm.sf(z) if alt == "greater" else stats.norm.cdf(z)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    return dict(n1=n1, n2=n2, x1=int(x1), x2=int(x2), est1=p1, est2=p2, dif=dif, pc=pc, stat=float(z), gl=None,
                p=float(p), ic=(dif - zc * se, dif + zc * se), se=se)


def welch(ya: np.ndarray, yb: np.ndarray, alt: str, conf: float = CONFIANCA) -> dict:
    n1, n2 = len(ya), len(yb)
    m1, m2 = ya.mean(), yb.mean()
    v1, v2 = ya.var(ddof=1), yb.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    dif = m1 - m2
    t = dif / se
    gl = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = stats.t.sf(t, gl) if alt == "greater" else stats.t.cdf(t, gl)
    tc = stats.t.ppf(1 - (1 - conf) / 2, gl)
    return dict(n1=n1, n2=n2, est1=m1, est2=m2, dif=dif, stat=float(t), gl=float(gl), p=float(p),
                ic=(dif - tc * se, dif + tc * se), se=se, dp1=ya.std(ddof=1), dp2=yb.std(ddof=1),
                assim1=float(stats.skew(ya)), assim2=float(stats.skew(yb)))


def executar(h: dict, ya: np.ndarray, yb: np.ndarray) -> dict:
    if h["tipo"] == "prop":
        return duas_proporcoes(int(ya.sum()), len(ya), int(yb.sum()), len(yb), h["alt"])
    return welch(ya, yb, h["alt"])


def diferenca_populacao(h: dict, v: dict) -> dict:
    """Parâmetros verdadeiros: valor em cada grupo e diferença A − B na população inteira."""
    ya, yb = grupos(h, v)
    return dict(est1=float(ya.mean()), est2=float(yb.mean()), dif=float(ya.mean() - yb.mean()), n1=len(ya), n2=len(yb))


def h1_verdadeira(h: dict, dif: float) -> bool:
    return dif > 0 if h["alt"] == "greater" else dif < 0


def exibir(h: dict, d):
    """Diferença na unidade mostrada ao leitor: pontos percentuais (proporções) ou % da média geométrica (salários)."""
    d = np.asarray(d, dtype=float)
    return 100 * d if h["tipo"] == "prop" else 100 * (10**d - 1)


# ----------------------------------------------------------------------------------------------
# variabilidade amostral: muitas amostras do mesmo tamanho, as mesmas hipóteses
# ----------------------------------------------------------------------------------------------
def simular(v: dict, n: int, B: int = N_SIMULACOES, semente: int = SEMENTE_SIMULACAO) -> dict:
    """Sorteia B amostras AAS de tamanho n, roda os cinco testes em cada uma e guarda diferença, rejeição e cobertura."""
    N = len(v["adm"])
    rng = np.random.default_rng(semente)
    verdade = {h["id"]: diferenca_populacao(h, v)["dif"] for h in HIPOTESES}
    res = {h["id"]: dict(dif=np.empty(B), est1=np.empty(B), est2=np.empty(B), rejeita=np.empty(B, bool), cobre=np.empty(B, bool))
           for h in HIPOTESES}
    for i in range(B):
        idx = rng.choice(N, size=n, replace=False)
        for h in HIPOTESES:
            r = executar(h, *grupos(h, v, idx))
            res[h["id"]]["dif"][i] = r["dif"]
            res[h["id"]]["est1"][i] = r["est1"]
            res[h["id"]]["est2"][i] = r["est2"]
            res[h["id"]]["rejeita"][i] = r["p"] < ALFA
            res[h["id"]]["cobre"][i] = r["ic"][0] <= verdade[h["id"]] <= r["ic"][1]
    return dict(B=B, n=n, semente=semente, verdade=verdade, res=res)

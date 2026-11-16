"""Aba "Informações extras": documentação de como a base, a população e a amostra foram construídas."""
import numpy as np
import streamlit as st
from scipy import stats

import inferencia as inf
from aba_inferencia import fig_qq, md, tabela_condicoes
from aba_populacao import DICIONARIO
from amostragem import amostra_oficial, carregar_meta_amostra, tamanho_amostra
from comum import FONTE_PASTA, FONTE_URL, MES_ROTULO, REPO_URL, brl, carregar_log, carregar_meta, n_br, pct


def secao_populacao(meta: dict):
    st.subheader("Como a população foi definida")
    n_clean, n_raw = meta["linhas_clean"], meta["linhas_raw"]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"""
**Fonte:** Novo CAGED (Cadastro Geral de Empregados e Desempregados), do Ministério do Trabalho e Emprego.
Microdados de movimentações (arquivo `CAGEDMOV202609`), competência **{MES_ROTULO}**.

**Link do arquivo original:** [{FONTE_URL}]({FONTE_URL}) (pasta: [{FONTE_PASTA}]({FONTE_PASTA})).

**Unidade de observação:** cada linha é **uma movimentação**, isto é, uma admissão ou um desligamento de
emprego formal (CLT) declarado ao governo no prazo do mês. **Não é uma pessoa** e não há identificador de
pessoa nem de empresa: quem foi contratado antes e continua empregado não aparece.

**População (N):** **{n_br(n_clean)}** movimentações, depois da limpeza ({n_br(n_raw)} no arquivo original; {n_br(meta['pct_mantido'], 2)}% mantidos).

**O que não está nela:** trabalho informal, servidores estatutários, e declarações feitas fora do prazo
ou excluídas depois (arquivos `CAGEDFOR` e `CAGEDEXC`, que não usamos).
"""
        )
    with c2:
        with st.container(border=True):
            st.markdown("**Como cada linha foi tratada**")
            st.markdown(
                "- Saem linhas sem escala mensal de salário (pagamento por tarefa, variável ou não identificado).\n"
                "- Saem salários ou valores iguais ou menores que zero.\n"
                "- Salário por hora que na verdade é salário mensal digitado no campo errado é **corrigido**.\n"
                "- Saem salários absurdos para a ocupação (acima de 100 vezes a mediana dela). **Não há teto fixo de salário.**\n"
                "- Idade ausente é preenchida pela mediana da mesma categoria de trabalhador."
            )
    with st.expander("Variáveis analisadas (tipo e significado)"):
        st.dataframe(DICIONARIO, hide_index=True, width="stretch")
    with st.expander("Log da limpeza (quantas linhas cada regra afetou)"):
        log = carregar_log().rename(columns={"regra": "Regra", "acao": "Ação", "linhas_afetadas": "Linhas afetadas",
                                             "pct_do_raw": "% do original", "linhas_restantes": "Linhas restantes"})
        st.dataframe(log, hide_index=True, width="stretch")
        st.caption(f"SHA-256 do arquivo original: `{meta['sha256_original']}`. O salário mensal que calculamos "
                   f"coincide em {meta['concordancia_salario_ministerio_pct']:.0f}% das linhas com o salário informado pelo Ministério.")


def secao_amostra(N: int, m: dict):
    st.subheader("Como a amostra foi dimensionada e sorteada")
    t = tamanho_amostra(m["confianca"], m["margem_erro"], m["p"], N)
    j = m["justificativas"]
    st.markdown(
        f"""
**Tamanho.** Para estimar uma proporção com a margem de erro desejada, usa-se
n = z² · p · (1 − p) / e². Com confiança de {pct(m['confianca'], 0)} (z = {n_br(t['z'], 2)}), margem de erro de
{n_br(100 * m['margem_erro'], 0)} pontos percentuais (e = {n_br(m['margem_erro'], 2)}) e p = {n_br(m['p'], 1)}:
n = {n_br(t['z'], 2)}² × {n_br(m['p'], 1)} × {n_br(1 - m['p'], 1)} / {n_br(m['margem_erro'], 2)}² = **{n_br(t['n0'], 2)}**,
arredondado para cima: **n = {n_br(t['n'])}**. O valor sai do código (`app/amostragem.py`), nunca é digitado,
e pode ser conferido na calculadora de tamanho de amostra da USP/Bauru com os mesmos parâmetros.

**Por que esses parâmetros.**
- *Confiança:* {j['confianca']}
- *Margem de erro:* {j['margem_erro']}
- *Proporção esperada (p):* {j['p']}

**Correção para população finita.** Só se aplica quando a amostra passa de 5% da população. Aqui ela é
{pct(m['fracao_amostral_pct'] / 100, 3)} de N = {n_br(N)}, então a correção é desprezível e **não foi aplicada**
(aplicando-a, n seria {n_br(m['n_se_corrigisse_populacao_finita'])}).

**Sorteio.** Amostragem aleatória simples (AAS) **sem reposição**: cada conjunto de {n_br(t['n'])} movimentações
tinha a mesma chance de ser escolhido, e nenhuma aparece duas vezes. A amostra é **uma só e fixa**: foi sorteada uma
vez, com a semente **{m['semente']}** (definida antes de ver qualquer resultado), e todas as abas usam essa mesma amostra.
"""
    )
    with st.expander("Detalhes técnicos para reproduzir o sorteio"):
        st.markdown(
            f"""
- **Método:** `{m['metodo']}`, com NumPy {m['numpy']}.
- **Semente:** `{m['semente']}`.
- **Identificação da linha:** a coluna `id_populacao` do arquivo da amostra é a posição (a partir de 0) da movimentação em `app/dados/populacao.parquet`.
- **SHA-256 da população usada no sorteio:** `{m['sha256_populacao']}`
- **Gerado em:** {m['gerado_em'].replace('T', ' ')}.
- **Código:** `scripts/sorteio_amostra.py` (grava `dados/amostra/amostra_aas.csv` e `amostra_meta.json`).
"""
        )


def secao_inferencia(am, m: dict, n_pop: int):
    st.subheader("Como os testes de hipóteses foram feitos")
    t = tamanho_amostra(m["confianca"], m["margem_erro"], m["p"], n_pop)
    st.markdown(md(
        f"Todos os testes da aba Inferência usam a **amostra oficial** (n = {n_br(len(am))} de {n_br(n_pop)} movimentações). "
        "Cada um compara dois grupos de movimentações independentes. A pergunta e a **direção** de H1 foram definidas "
        "antes de olhar os resultados. O valor-p é **unilateral**, na direção de H1, e o intervalo de confiança é **bilateral**. "
        "Os salários são analisados em **log10** (comparação de médias geométricas), porque a distribuição bruta tem cauda longa à direita."
    ))
    st.markdown("**Confiança, significância e margem de erro**")
    st.markdown(md(
        f"- **Confiança e significância são o mesmo número visto de dois lados:** α = 1 − confiança. Usamos confiança de "
        f"{pct(inf.CONFIANCA, 0)} no dimensionamento da amostra e nos intervalos, e α = {pct(inf.ALFA, 0)} nos testes.\n"
        f"- **A margem de erro de {pct(m['margem_erro'], 0)} é outra coisa:** serviu só para calcular o tamanho da amostra "
        f"(n = {n_br(t['n'])}). Ela é a precisão da estimativa de **uma** proporção com a amostra inteira, no pior caso (p = 0,5). "
        "Nos testes, cada grupo usa só uma parte da amostra e comparamos **diferenças** entre dois grupos, por isso os intervalos "
        "ficam mais largos (da ordem de ±4 a ±5 pontos percentuais nas taxas), e não ±2.\n"
        f"- **Teste unilateral e IC bilateral:** um teste unilateral a {pct(inf.ALFA, 0)} corresponde, em rigor, a um IC bilateral de "
        f"{pct(1 - 2 * inf.ALFA, 0)}. Usamos o IC bilateral de {pct(inf.CONFIANCA, 0)}, que é o padrão e mais conservador. Em casos limítrofes "
        "(valor-p entre 2,5% e 5%) o IC poderia incluir o zero e o teste rejeitar H0. Isso não acontece em nenhuma das cinco hipóteses."
    ))
    st.markdown("**Como ler a simulação de variabilidade amostral (aba Inferência, Resumo)**")
    st.markdown(md(
        "- Quando a H1 é **verdadeira**, a coluna \"amostras que rejeitam H0\" é o **poder** do teste. Poder baixo explica por que uma diferença real pode passar "
        "despercebida numa amostra.\n"
        "- Quando a H1 é **falsa**, essa coluna mostra com que frequência uma amostra levaria a uma conclusão errada.\n"
        "- Cerca de 95% dos intervalos de confiança devem conter o valor verdadeiro: é isso que significa \"95% de confiança\"."))
    with st.expander("Por que o salário é analisado em log10? (QQ plot do salário bruto e do log10)"):
        sal = am["salario"].to_numpy(dtype=float)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_qq(sal, "Salário bruto (R$)", False), width="stretch", key="qq_bruto")
        c2.plotly_chart(fig_qq(np.log10(sal), "Log10 do salário", True), width="stretch", key="qq_log")
        st.markdown(md(
            f"O salário bruto se afasta muito da reta (assimetria {n_br(stats.skew(sal), 1)}): poucos valores muito altos "
            f"puxam a média, que fica em {brl(sal.mean(), 0)} contra mediana de {brl(np.median(sal), 0)}. No log10 o miolo da "
            "distribuição fica próximo da reta. O que sobra de desvio está na ponta esquerda, onde há poucas movimentações com "
            "salário muito baixo (a amostra tem uma de R$ 0,01). A limpeza removeu salários absurdamente altos "
            "para a ocupação, mas não aplicou regra equivalente para salários baixos."))


def secao_condicoes(am, n_pop: int):
    st.subheader("Verificação das condições dos testes")
    st.markdown(md(
        "Para cada hipótese, as condições que sustentam o teste e como foram checadas na amostra oficial. "
        "Os QQ plots de cada hipótese estão na aba Inferência."))
    v = inf.vetores(am)
    for h in inf.HIPOTESES:
        r = inf.executar(h, *inf.grupos(h, v))
        st.markdown(f"**{h['id']}. {h['titulo']}**")
        st.table(tabela_condicoes(h, r, n_pop).set_index("Condição"))


def secao_arquivos():
    st.subheader("Código e arquivos do projeto")
    st.markdown(
        f"""
Repositório público: [{REPO_URL}]({REPO_URL}). O caminho dos dados até o painel é:

1. **Arquivo original** do Novo CAGED (link na seção acima), mantido sem alteração.
2. `scripts/limpeza.py`: aplica as regras de limpeza e grava a população tratada, o log e os metadados.
3. `scripts/gerar_dados_app.py`: reduz a população às 11 colunas usadas e grava `app/dados/populacao.parquet`.
4. `scripts/sorteio_amostra.py`: calcula n, sorteia a AAS e grava `app/dados/amostra_aas.csv` e `amostra_meta.json`.
5. `app/`: este painel (abas População, Amostra, Inferência e Informações extras).
"""
    )


def render(pop):
    am = amostra_oficial(pop)
    st.header("Informações extras")
    st.caption("Documentação do processo: de onde vêm os dados, como a população foi definida e como a amostra foi obtida.")
    secao_populacao(carregar_meta())
    st.divider()
    meta_am = carregar_meta_amostra()
    n_pop = carregar_meta()["linhas_clean"]
    secao_amostra(n_pop, meta_am)
    st.divider()
    secao_inferencia(am, meta_am, n_pop)
    st.divider()
    secao_condicoes(am, n_pop)
    st.divider()
    secao_arquivos()

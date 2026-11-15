"""Aba "Informações extras": documentação de como a base, a população e a amostra foram construídas."""
import streamlit as st

from aba_populacao import DICIONARIO
from amostragem import carregar_meta_amostra, tamanho_amostra
from comum import FONTE_PASTA, FONTE_URL, MES_ROTULO, REPO_URL, carregar_log, carregar_meta, n_br, pct


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


def render():
    st.header("Informações extras")
    st.caption("Documentação do processo: de onde vêm os dados, como a população foi definida e como a amostra foi obtida.")
    secao_populacao(carregar_meta())
    st.divider()
    secao_amostra(carregar_meta()["linhas_clean"], carregar_meta_amostra())
    st.divider()
    secao_arquivos()

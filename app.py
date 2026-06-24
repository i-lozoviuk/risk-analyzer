"""
AI Project Risk Analyzer.

Streamlit app with a 3-step flow:
1. User describes their project.
2. Claude identifies 5-7 key risks (JSON), user rates each risk's
   probability and criticality.
3. App renders a 3x3 risk matrix (matplotlib) plus a color-coded table.

The interface language (Russian/English/Ukrainian) and the analysis
language (the language Claude must answer in) are independent settings,
both selectable at the top of the page.
"""

import html
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from anthropic import Anthropic, APIError, APIConnectionError
from dotenv import load_dotenv
from matplotlib.lines import Line2D

load_dotenv()

MODEL_NAME = "claude-sonnet-4-6"

# Ordinal scale shared by AI severity and user ratings, low to high. These are
# internal canonical codes (always English); UI_LEVEL_LABELS translates them for display.
LEVELS = ["Low", "Medium", "High"]
LEVEL_TO_SCORE = {level: index + 1 for index, level in enumerate(LEVELS)}
LEVEL_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}

# Light backgrounds for the matrix cells, keyed by combined risk level.
CELL_BACKGROUND_COLORS = {"Low": "#A9DFBF", "Medium": "#F9E79F", "High": "#F5B7B1"}

# Row colors for the final summary table, keyed by combined risk level.
ROW_STYLES = {
    "High": ("#FF4444", "#FFFFFF"),
    "Medium": ("#FFA500", "#FFFFFF"),
    "Low": ("#2ECC71", "#FFFFFF"),
}

UI_LANGUAGE_OPTIONS = {"ru": "Русский", "en": "English", "uk": "Українська"}

ANALYSIS_LANGUAGE_OPTIONS = {
    "ru": "Русский",
    "en": "English",
    "uk": "Українська",
    "es": "Español",
    "de": "Deutsch",
}
ANALYSIS_LANGUAGE_PROMPT_NAMES = {
    "ru": "Russian",
    "en": "English",
    "uk": "Ukrainian",
    "es": "Spanish",
    "de": "German",
}

UI_LEVEL_LABELS = {
    "ru": {"Low": "Низкий", "Medium": "Средний", "High": "Высокий"},
    "en": {"Low": "Low", "Medium": "Medium", "High": "High"},
    "uk": {"Low": "Низький", "Medium": "Середній", "High": "Високий"},
}

UI_TRANSLATIONS = {
    "ru": {
        "language_selector_label": "Язык интерфейса",
        "app_title": "AI-анализатор рисков проекта",
        "app_description": (
            "Этот инструмент помогает предпринимателям выявить ключевые риски проекта до его "
            "запуска. Опишите свой проект — AI проанализирует его и предложит список рисков с "
            "оценкой опасности. Вы сможете уточнить вероятность и критичность каждого риска, "
            "после чего инструмент построит матрицу рисков и даст рекомендации."
        ),
        "example_expander_label": "Посмотреть пример описания",
        "example_description": (
            "Запускаю онлайн-магазин спортивного питания в Киеве. Бюджет $20,000, срок запуска "
            "3 месяца. Планирую закупать товар у поставщиков из Китая, запустить рекламу в "
            "Instagram и доставлять по всей Украине. Целевая аудитория — люди 25-40 лет, "
            "занимающиеся спортом. Команда: я и один менеджер по продажам. Ожидаемая выручка "
            "в первый месяц — $3,000."
        ),
        "use_example_button": "Использовать этот пример",
        "step1_header": "Шаг 1: Опишите свой проект",
        "project_description_label": "Описание проекта",
        "project_description_placeholder": "Опишите ваш проект, его цели, рынок и ключевую деятельность...",
        "analysis_language_label": "Язык анализа",
        "analysis_language_hint": "Описание проекта можно писать на любом языке — AI поймёт.",
        "identify_risks_button": "Выявить риски",
        "warning_empty_description": "Пожалуйста, сначала введите описание проекта.",
        "spinner_analyzing": "Анализ рисков проекта с помощью Claude...",
        "error_api": "Не удалось связаться с API Claude: {error}",
        "error_analysis": "Анализ рисков не удался: {error}",
        "step2_header": "Шаг 2: Просмотр и оценка рисков",
        "step2_caption": "Первоначальная оценка Claude показана для справки. Оцените каждый риск самостоятельно ниже.",
        "ai_severity_prefix": "ИИ:",
        "recommendation_prefix": "Рекомендация:",
        "probability_rating_label": "Ваша оценка вероятности",
        "criticality_rating_label": "Ваша оценка критичности",
        "generate_matrix_button": "Построить матрицу рисков",
        "step3_header": "Шаг 3: Матрица рисков и итоги",
        "table_col_number": "#",
        "table_col_risk": "Риск",
        "table_col_ai_severity": "Оценка ИИ",
        "table_col_your_probability": "Ваша вероятность",
        "table_col_your_criticality": "Ваша критичность",
        "table_col_risk_level": "Уровень риска",
        "table_col_recommendation": "Рекомендация",
    },
    "en": {
        "language_selector_label": "Interface language",
        "app_title": "AI Project Risk Analyzer",
        "app_description": (
            "This tool helps entrepreneurs identify key project risks before launch. Describe "
            "your project — AI will analyze it and suggest a list of risks with a danger rating. "
            "You'll be able to refine the probability and criticality of each risk, after which "
            "the tool will build a risk matrix and give recommendations."
        ),
        "example_expander_label": "View example description",
        "example_description": (
            "Launching an online sports nutrition store in Kyiv. Budget $20,000, launch timeline "
            "3 months. Planning to source products from suppliers in China, run Instagram ads, "
            "and deliver across Ukraine. Target audience — people aged 25-40 who are into sports. "
            "Team: myself and one sales manager. Expected revenue in the first month — $3,000."
        ),
        "use_example_button": "Use this example",
        "step1_header": "Step 1: Describe Your Project",
        "project_description_label": "Project description",
        "project_description_placeholder": "Describe your project, its goals, market, and key activities...",
        "analysis_language_label": "Analysis language",
        "analysis_language_hint": "You can write the project description in any language — AI will understand.",
        "identify_risks_button": "Identify Risks",
        "warning_empty_description": "Please enter a project description first.",
        "spinner_analyzing": "Analyzing project risks with Claude...",
        "error_api": "Could not reach the Claude API: {error}",
        "error_analysis": "Risk analysis failed: {error}",
        "step2_header": "Step 2: Review and Rate Risks",
        "step2_caption": "Claude's initial severity is shown for reference. Rate each risk yourself below.",
        "ai_severity_prefix": "AI:",
        "recommendation_prefix": "Recommendation:",
        "probability_rating_label": "Your probability rating",
        "criticality_rating_label": "Your criticality rating",
        "generate_matrix_button": "Generate Risk Matrix",
        "step3_header": "Step 3: Risk Matrix and Summary",
        "table_col_number": "#",
        "table_col_risk": "Risk",
        "table_col_ai_severity": "AI Severity",
        "table_col_your_probability": "Your Probability",
        "table_col_your_criticality": "Your Criticality",
        "table_col_risk_level": "Risk Level",
        "table_col_recommendation": "Recommendation",
    },
    "uk": {
        "language_selector_label": "Мова інтерфейсу",
        "app_title": "AI-аналізатор ризиків проєкту",
        "app_description": (
            "Цей інструмент допомагає підприємцям виявити ключові ризики проєкту до його "
            "запуску. Опишіть свій проєкт — AI проаналізує його і запропонує список ризиків з "
            "оцінкою небезпечності. Ви зможете уточнити ймовірність та критичність кожного "
            "ризику, після чого інструмент побудує матрицю ризиків і надасть рекомендації."
        ),
        "example_expander_label": "Переглянути приклад опису",
        "example_description": (
            "Запускаю онлайн-магазин спортивного харчування в Києві. Бюджет $20,000, термін "
            "запуску 3 місяці. Планую закупати товар у постачальників з Китаю, запустити "
            "рекламу в Instagram і доставляти по всій Україні. Цільова аудиторія — люди 25-40 "
            "років, які займаються спортом. Команда: я і один менеджер з продажу. Очікувана "
            "виручка в перший місяць — $3,000."
        ),
        "use_example_button": "Використати цей приклад",
        "step1_header": "Крок 1: Опишіть свій проєкт",
        "project_description_label": "Опис проєкту",
        "project_description_placeholder": "Опишіть свій проєкт, його цілі, ринок і ключову діяльність...",
        "analysis_language_label": "Мова аналізу",
        "analysis_language_hint": "Опис проєкту можна писати будь-якою мовою — AI зрозуміє.",
        "identify_risks_button": "Виявити ризики",
        "warning_empty_description": "Будь ласка, спочатку введіть опис проєкту.",
        "spinner_analyzing": "Аналіз ризиків проєкту за допомогою Claude...",
        "error_api": "Не вдалося зв'язатися з API Claude: {error}",
        "error_analysis": "Аналіз ризиків не вдався: {error}",
        "step2_header": "Крок 2: Перегляд та оцінка ризиків",
        "step2_caption": "Початкова оцінка Claude показана для довідки. Оцініть кожен ризик самостійно нижче.",
        "ai_severity_prefix": "ШІ:",
        "recommendation_prefix": "Рекомендація:",
        "probability_rating_label": "Ваша оцінка ймовірності",
        "criticality_rating_label": "Ваша оцінка критичності",
        "generate_matrix_button": "Побудувати матрицю ризиків",
        "step3_header": "Крок 3: Матриця ризиків та підсумки",
        "table_col_number": "#",
        "table_col_risk": "Ризик",
        "table_col_ai_severity": "Оцінка ШІ",
        "table_col_your_probability": "Ваша ймовірність",
        "table_col_your_criticality": "Ваша критичність",
        "table_col_risk_level": "Рівень ризику",
        "table_col_recommendation": "Рекомендація",
    },
}

SYSTEM_PROMPT_BASE = (
    "You are a senior risk management consultant. Given a project "
    "description, identify the 5 to 7 most important risks. "
    "Respond with ONLY a raw JSON array (no markdown, no commentary). "
    "Each array item must be an object with exactly these keys: "
    '"name" (short risk title), "description" (1-2 sentences), '
    '"severity" (one of the English words "Low", "Medium", "High" - do not '
    'translate this field), and "recommendation" (1-2 sentences of mitigation advice).'
)


def get_ui_language() -> str:
    return st.session_state.get("ui_language", "ru")


def tr(key: str, **kwargs) -> str:
    """Look up a UI string in the current interface language."""
    template = UI_TRANSLATIONS[get_ui_language()][key]
    return template.format(**kwargs) if kwargs else template


def level_label(level: str) -> str:
    """Translate a canonical Low/Medium/High level into the current interface language."""
    return UI_LEVEL_LABELS[get_ui_language()][level]


def get_client() -> Anthropic:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to a .env file or Streamlit Secrets."
        )
    return Anthropic(api_key=api_key)


def extract_json_array(raw_text: str) -> str:
    """Strip optional markdown code fences Claude may add around the JSON."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


def normalize_text(text: str) -> str:
    """Fix stray unicode whitespace and missing spaces that make sentences run together."""
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"\s+", " ", text).strip()
    # LLM JSON generation occasionally drops the space after sentence punctuation,
    # e.g. "...beans.Dependence on..." - reinsert it before a following capital letter.
    text = re.sub(r"([.!?,;:])(?=[A-Z])", r"\1 ", text)
    return text


def escape_markdown_dollar(text: str) -> str:
    """Escape literal $ so Streamlit's markdown renderer doesn't treat it as a LaTeX delimiter.

    Two unescaped $ in the same st.write()/st.markdown() call (e.g. "$20,000 ... $3,000")
    get parsed as inline math - everything between them renders as KaTeX, which drops the
    normal word spacing and looks like the text "ran together".
    """
    return text.replace("$", "\\$")


def prepare_html_cell(text: str) -> str:
    """Make AI-generated text safe to embed in a raw HTML table cell.

    HTML-escapes the text (it is rendered via unsafe_allow_html=True), converts any
    embedded newline into an explicit <br>, and escapes $ to avoid the same KaTeX
    word-squishing issue described in escape_markdown_dollar.
    """
    escaped = html.escape(text)
    escaped = escaped.replace("\n", "<br>")
    escaped = escaped.replace("$", "\\$")
    return escaped


def build_system_prompt(analysis_language: str) -> str:
    language_name = ANALYSIS_LANGUAGE_PROMPT_NAMES[analysis_language]
    return (
        f"{SYSTEM_PROMPT_BASE} Respond strictly in {language_name}. All risk names, "
        "descriptions and recommendations must be in this language regardless of the "
        "input language."
    )


def call_claude_for_risks(project_description: str, analysis_language: str) -> list[dict]:
    """Ask Claude for a structured list of project risks. Raises on failure."""
    client = get_client()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        system=build_system_prompt(analysis_language),
        messages=[{"role": "user", "content": project_description}],
    )
    raw_text = response.content[0].text
    json_text = extract_json_array(raw_text)
    risks = json.loads(json_text)

    for risk in risks:
        if risk.get("severity") not in LEVELS:
            risk["severity"] = "Medium"
        for field in ("name", "description", "recommendation"):
            if field in risk:
                risk[field] = normalize_text(risk[field])
    return risks


def level_from_combined_score(combined_score: int) -> str:
    """Map a probability+criticality score (2-6) to a Low/Medium/High risk level."""
    if combined_score <= 3:
        return "Low"
    if combined_score <= 5:
        return "Medium"
    return "High"


def combined_risk_level(probability: str, criticality: str) -> str:
    return level_from_combined_score(LEVEL_TO_SCORE[probability] + LEVEL_TO_SCORE[criticality])


def init_session_state() -> None:
    if "risks" not in st.session_state:
        st.session_state.risks = None  # list of risk dicts once Step 2 succeeds
    if "matrix_generated" not in st.session_state:
        st.session_state.matrix_generated = False


def reset_results() -> None:
    """Discard any previously generated risks - used when the analysis language changes."""
    st.session_state.risks = None
    st.session_state.matrix_generated = False


def use_example_description() -> None:
    st.session_state.project_description = tr("example_description")


def render_header() -> None:
    title_col, language_col = st.columns([3, 1])
    with title_col:
        st.title(tr("app_title"))
    with language_col:
        st.selectbox(
            tr("language_selector_label"),
            options=list(UI_LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: UI_LANGUAGE_OPTIONS[code],
            key="ui_language",
        )

    st.info(tr("app_description"))

    with st.expander(tr("example_expander_label")):
        st.write(escape_markdown_dollar(tr("example_description")))
        st.button(tr("use_example_button"), on_click=use_example_description)


def render_step1() -> str | None:
    st.header(tr("step1_header"))

    text_col, lang_col = st.columns([3, 1])
    with text_col:
        description = st.text_area(
            tr("project_description_label"),
            height=150,
            placeholder=tr("project_description_placeholder"),
            key="project_description",
        )
    with lang_col:
        st.selectbox(
            tr("analysis_language_label"),
            options=list(ANALYSIS_LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: ANALYSIS_LANGUAGE_OPTIONS[code],
            key="analysis_language",
            on_change=reset_results,
        )
        st.caption(tr("analysis_language_hint"))

    if st.button(tr("identify_risks_button"), type="primary"):
        if not description.strip():
            st.warning(tr("warning_empty_description"))
            return None
        return description
    return None


def render_step2() -> None:
    st.header(tr("step2_header"))
    st.caption(tr("step2_caption"))

    for index, risk in enumerate(st.session_state.risks):
        with st.container(border=True):
            col_title, col_badge = st.columns([4, 1])
            with col_title:
                st.subheader(f"{index + 1}. {escape_markdown_dollar(risk['name'])}")
            with col_badge:
                st.markdown(
                    f"<span style='color:{LEVEL_COLORS[risk['severity']]}; "
                    f"font-weight:bold;'>{tr('ai_severity_prefix')} {level_label(risk['severity'])}</span>",
                    unsafe_allow_html=True,
                )

            st.write(escape_markdown_dollar(risk["description"]))
            st.info(f"{tr('recommendation_prefix')} {escape_markdown_dollar(risk['recommendation'])}")

            rating_col1, rating_col2 = st.columns(2)
            with rating_col1:
                st.radio(
                    tr("probability_rating_label"),
                    LEVELS,
                    index=1,
                    horizontal=True,
                    format_func=level_label,
                    key=f"probability_{index}",
                )
            with rating_col2:
                st.radio(
                    tr("criticality_rating_label"),
                    LEVELS,
                    index=1,
                    horizontal=True,
                    format_func=level_label,
                    key=f"criticality_{index}",
                )

    st.button(
        tr("generate_matrix_button"),
        type="primary",
        on_click=lambda: st.session_state.update(matrix_generated=True),
    )


def collect_user_ratings() -> list[dict]:
    """Combine AI risk data with the user's rating widgets into one structure."""
    combined = []
    for index, risk in enumerate(st.session_state.risks):
        combined.append(
            {
                "name": risk["name"],
                "description": risk["description"],
                "ai_severity": risk["severity"],
                "recommendation": risk["recommendation"],
                "user_probability": st.session_state[f"probability_{index}"],
                "user_criticality": st.session_state[f"criticality_{index}"],
            }
        )
    return combined


def build_risk_matrix_figure(rated_risks: list[dict]) -> plt.Figure:
    """Render a 3x3 probability/criticality matrix with each risk plotted as a labeled point."""
    fig, ax = plt.subplots(figsize=(7.5, 9.5))

    # Color the background cells from green (low risk) to red (high risk) and label each cell.
    for prob_score in range(1, 4):
        for crit_score in range(1, 4):
            level = level_from_combined_score(prob_score + crit_score)
            ax.add_patch(
                plt.Rectangle(
                    (prob_score - 1.5, crit_score - 1.5), 1, 1,
                    color=CELL_BACKGROUND_COLORS[level], zorder=0,
                )
            )
            ax.text(
                prob_score, crit_score - 0.38, f"{level} Risk",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#555555", zorder=1,
            )

    # Group risks by grid cell so overlapping points get a small offset.
    cell_occupants: dict[tuple[int, int], int] = {}
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for risk_number, risk in enumerate(rated_risks, start=1):
        prob_score = LEVEL_TO_SCORE[risk["user_probability"]]
        crit_score = LEVEL_TO_SCORE[risk["user_criticality"]]
        cell_key = (prob_score, crit_score)
        occupant_index = cell_occupants.get(cell_key, 0)
        cell_occupants[cell_key] = occupant_index + 1

        offset = (occupant_index - 1) * 0.18
        x = prob_score + offset
        y = crit_score + offset

        point_color = LEVEL_COLORS[risk["ai_severity"]]
        ax.scatter(x, y, s=500, color=point_color, edgecolors="black", linewidths=1.5, zorder=2)
        ax.annotate(
            str(risk_number), (x, y), ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=3,
        )

        legend_handles.append(
            Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor=point_color, markeredgecolor="black")
        )
        legend_labels.append(
            f"{risk_number}: {escape_markdown_dollar(risk['name'])} "
            f"(Probability={risk['user_probability']}, Criticality={risk['user_criticality']})"
        )

    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(LEVELS, fontsize=12)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(LEVELS, fontsize=12)
    ax.set_xlabel("Probability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Criticality", fontsize=14, fontweight="bold")
    ax.set_title("Risk Matrix", fontsize=16, fontweight="bold")
    ax.grid(False)

    ax.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        fontsize=9, title="Risks", title_fontsize=10, frameon=True,
    )
    fig.subplots_adjust(bottom=0.05 + 0.035 * len(rated_risks))
    return fig


def render_styled_risk_table(rated_risks: list[dict]) -> str:
    """Build an HTML table with header/row styling that st.dataframe cannot express."""
    canonical_levels: list[str] = []
    rows = []
    for index, risk in enumerate(rated_risks):
        risk_level = combined_risk_level(risk["user_probability"], risk["user_criticality"])
        canonical_levels.append(risk_level)
        rows.append(
            {
                tr("table_col_number"): index + 1,
                tr("table_col_risk"): prepare_html_cell(risk["name"]),
                tr("table_col_ai_severity"): level_label(risk["ai_severity"]),
                tr("table_col_your_probability"): level_label(risk["user_probability"]),
                tr("table_col_your_criticality"): level_label(risk["user_criticality"]),
                tr("table_col_risk_level"): level_label(risk_level),
                tr("table_col_recommendation"): prepare_html_cell(risk["recommendation"]),
            }
        )
    table_data = pd.DataFrame(rows)

    def style_row(row: pd.Series) -> list[str]:
        bg_color, text_color = ROW_STYLES[canonical_levels[row.name]]
        return [f"background-color: {bg_color}; color: {text_color};"] * len(row)

    styler = (
        table_data.style.apply(style_row, axis=1).hide(axis="index").set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#2C3E50"),
                        ("color", "#FFFFFF"),
                        ("font-weight", "bold"),
                        ("font-size", "14px"),
                        ("border", "1px solid #34495E"),
                        ("padding", "8px"),
                        ("text-align", "left"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "14px"),
                        ("border", "1px solid #34495E"),
                        ("padding", "8px"),
                    ],
                },
                {
                    "selector": "table",
                    "props": [("border-collapse", "collapse"), ("width", "100%")],
                },
            ]
        )
    )
    return styler.to_html()


def render_step3(rated_risks: list[dict]) -> None:
    st.header(tr("step3_header"))

    fig = build_risk_matrix_figure(rated_risks)
    st.pyplot(fig)

    st.markdown(render_styled_risk_table(rated_risks), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="AI Project Risk Analyzer", layout="centered")
    init_session_state()

    render_header()

    if st.session_state.risks is None:
        description = render_step1()
        if description:
            analysis_language = st.session_state.get("analysis_language", "ru")
            with st.spinner(tr("spinner_analyzing")):
                try:
                    st.session_state.risks = call_claude_for_risks(description, analysis_language)
                except (APIError, APIConnectionError) as error:
                    st.error(tr("error_api", error=error))
                except (RuntimeError, json.JSONDecodeError) as error:
                    st.error(tr("error_analysis", error=error))
            if st.session_state.risks:
                st.rerun()
        return

    render_step2()

    if st.session_state.matrix_generated:
        rated_risks = collect_user_ratings()
        render_step3(rated_risks)


if __name__ == "__main__":
    main()

"""Многоагентный сценарий: профиль из input.txt → три Markdown-отчёта в output/.

Запуск (окружение micromamba `deep`):
    micromamba activate deep
    export HF_TOKEN=...   # или huggingface-cli login
    python full_agent.py
    python full_agent.py --input ./input.txt --out ./output
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from huggingface_hub import get_token
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel


@dataclass(frozen=True)
class Profile:
    competencies: str
    interests: str


def parse_input_txt(path: Path) -> Profile:
    raw = path.read_text(encoding="utf-8").lstrip("\ufeff")
    comp: list[str] = []
    intr: list[str] = []
    mode: str | None = None
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("компетенции"):
            mode = "c"
            tail = s.split(":", 1)[1].strip() if ":" in s else ""
            if tail:
                comp.append(tail)
        elif low.startswith("интересы"):
            mode = "i"
            tail = s.split(":", 1)[1].strip() if ":" in s else ""
            if tail:
                intr.append(tail)
        elif mode == "c":
            comp.append(s)
        elif mode == "i":
            intr.append(s)
    competencies = " ".join(comp).strip()
    interests = " ".join(intr).strip()
    if not competencies or not interests:
        raise ValueError(
            f"В {path} должны быть непустые блоки «Компетенции:» и «Интересы:»."
        )
    return Profile(competencies=competencies, interests=interests)


def build_model(model_id: str) -> InferenceClientModel:
    return InferenceClientModel(model_id=model_id)


def doc_preamble(
    *,
    title: str,
    agent_name: str,
    agent_role: str,
    how_produced: str,
    sources_note: str,
) -> str:
    today = date.today().isoformat()
    return f"""# {title}

## Как получен этот документ

- **Дата:** {today}
- **Пайплайн:** `full_agent.py` — несколько отдельных агентов с общей языковой моделью.
- **Агент:** «{agent_name}» — {agent_role}
- **Производство текста:** {how_produced}
- **Про достоверность:** {sources_note}

---

"""


def save_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def research_agent(model: InferenceClientModel) -> CodeAgent:
    return CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=True,
        name="market_analyst",
        max_steps=18,
        instructions=(
            "Ты аналитик рынка труда (акцент: Россия и смежные удалённые форматы). "
            "Обязательно используй поиск (duckduckgo_search), чтобы опереться на свежие страницы. "
            "В ответе указывай конкретные формулировки из найденных источников и по возможности URL. "
            "Не выдумывай названия вакансий и зарплат — если данных нет, так и напиши."
        ),
    )


def planner_agent(model: InferenceClientModel) -> CodeAgent:
    return CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=True,
        name="learning_planner",
        max_steps=14,
        instructions=(
            "Ты методист по обучению взрослых. Строишь реалистичный пошаговый план с учётом времени и ресурсов. "
            "Можешь слегка уточнить тренды поиском, но опирайся в первую очередь на переданный контекст исследования рынка."
        ),
    )


def ai_coach_agent(model: InferenceClientModel) -> CodeAgent:
    return CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=True,
        name="ai_learning_coach",
        max_steps=12,
        instructions=(
            "Ты наставник, который объясняет, как учиться с помощью ИИ-инструментов (чаты, ассистенты в IDE, RAG, генерация карточек). "
            "Давай конкретные приёмы, примеры промптов и предостережения (галлюцинации, приватность). "
            "Поиск используй выборочно для актуальных названий сервисов или статей."
        ),
    )


def run_pipeline(profile: Profile, out_dir: Path, model_id: str) -> None:
    model = build_model(model_id)
    comp_block = profile.competencies
    int_block = profile.interests

    print("— Этап 1/3: тренды, востребованность, вакансии…", flush=True)
    ra = research_agent(model)
    research_task = f"""
Профиль человека:
- Компетенции: {comp_block}
- Интересы: {int_block}

Документ по теме **«Тренды, что востребовано сейчас, какие вакансии»**. Обязательная структура (заголовки ##):

## Тренды: что востребовано сейчас
## Навыки и роли в спросе
## Какие вакансии встречаются (примеры из поиска с URL или цитатами; не выдумывать)
## Где искать вакансии и обновлять картину рынка
## Краткое резюме
## Ограничения данных (что проверить вручную)

Пиши на русском. Используй веб-поиск по ключевым словам из профиля и смежным IT/HR/образовательным запросам.
"""
    research_body = ra.run(research_task)
    research_md = (
        doc_preamble(
            title="Тренды, что востребовано сейчас, какие вакансии",
            agent_name="Аналитик рынка",
            agent_role="собирает картину рынка через веб-поиск и оформляет выводы.",
            how_produced="языковая модель + инструмент `DuckDuckGoSearchTool` (результаты поиска могут быть неполными).",
            sources_note="проверяй ссылки и формулировки вакансий на первоисточниках; модель может неточно интерпретировать фрагменты страниц.",
        )
        + str(research_body)
    )
    p1 = out_dir / "01_trendy_rynok_i_vakansii.md"
    save_markdown(p1, research_md)
    print(f"  сохранено: {p1}", flush=True)

    research_excerpt = str(research_body)
    if len(research_excerpt) > 14000:
        research_excerpt = research_excerpt[:14000] + "\n\n…[фрагмент обрезан для контекста следующего агента]…"

    print("— Этап 2/3: план обучения…", flush=True)
    pa = planner_agent(model)
    plan_task = f"""
Профиль:
- Компетенции: {comp_block}
- Интересы: {int_block}

Контекст исследования рынка (для согласования плана с реальным спросом):

{research_excerpt}

Документ по теме **«Пошаговый план обучения»**. Обязательная структура (##):

## Пошаговый план обучения (по неделям, минимум 8 недель; шаги нумеруй)
## Цели на 3 и 6 месяцев (кратко)
## Практика и портфолио
## Ресурсы (книги, курсы, сообщества) — пометки «бесплатно» / «платно», без выдуманных цен
## Как измерять прогресс
## Риски и как их снизить

Будь приземлённым: человек совмещает текущие компетенции с уклоном в указанные интересы.
"""
    plan_body = pa.run(plan_task)
    plan_md = (
        doc_preamble(
            title="Пошаговый план обучения",
            agent_name="Методист",
            agent_role="строит учебную траекторию с учётом профиля и краткого резюме рынка.",
            how_produced="языковая модель; опционально лёгкий фактчек через поиск.",
            sources_note="сроки и нагрузка — ориентиры; подстрой под своё расписание.",
        )
        + str(plan_body)
    )
    p2 = out_dir / "02_plan_obucheniya.md"
    save_markdown(p2, plan_md)
    print(f"  сохранено: {p2}", flush=True)

    plan_excerpt = str(plan_body)
    if len(plan_excerpt) > 6000:
        plan_excerpt = plan_excerpt[:6000] + "\n\n…[краткий фрагмент плана для контекста]…"

    print("— Этап 3/3: обучение с ИИ…", flush=True)
    aa = ai_coach_agent(model)
    ai_task = f"""
Профиль:
- Компетенции: {comp_block}
- Интересы: {int_block}

Фрагмент учебного плана (для согласования советов):

{plan_excerpt}

Документ по теме **«Само обучение с ИИ»**. Обязательная структура (##):

## Само обучение с ИИ: зачем это работает и ограничения
## Инструменты: чаты, IDE с LLM, заметки, карточки — когда что уместно
## Шаблоны промптов под задачи из плана (объяснение + примеры)
## Как проверять ответы ИИ (чек-лист)
## Приватность и этика (учебные данные, NDA)
## Недельная схема работы с ИИ под этот план

Тон: практичный, без маркетинговых обещаний.
"""
    ai_body = aa.run(ai_task)
    ai_md = (
        doc_preamble(
            title="Само обучение с ИИ",
            agent_name="Наставник по обучению с ИИ",
            agent_role="описывает практику использования ИИ в учёбе.",
            how_produced="языковая модель + при необходимости веб-поиск.",
            sources_note="названия продуктов и интерфейсы меняются — сверяйся с официальной документацией.",
        )
        + str(ai_body)
    )
    p3 = out_dir / "03_samoobuchenie_s_ii.md"
    save_markdown(p3, ai_md)
    print(f"  сохранено: {p3}", flush=True)

    index = f"""# Результаты пайплайна `full_agent.py`

Сгенерировано: **{date.today().isoformat()}**

| Файл | Содержание |
|------|------------|
| [01_trendy_rynok_i_vakansii.md](01_trendy_rynok_i_vakansii.md) | Тренды, что востребовано сейчас, какие вакансии |
| [02_plan_obucheniya.md](02_plan_obucheniya.md) | Пошаговый план обучения |
| [03_samoobuchenie_s_ii.md](03_samoobuchenie_s_ii.md) | Само обучение с ИИ |

## Как это устроено

Три **отдельных** `CodeAgent` в smolagents: аналитик рынка (упор на поиск), методист (план), наставник по ИИ (методики и промпты).
У каждого агент своё системное описание роли (`instructions`), у аналитика и остальных при необходимости вызывается `DuckDuckGoSearchTool`.

Исходный профиль читается из `input.txt` (блоки «Компетенции» и «Интересы»).

## Важно

Это **не** карьерная или юридическая консультация. Перед решениями о работе и обучении перепроверяйте факты и условия на первоисточниках.
"""
    save_markdown(out_dir / "README.md", index)
    print(f"  оглавление: {out_dir / 'README.md'}", flush=True)


def _require_ddgs() -> None:
    try:
        import ddgs  # noqa: F401
    except ImportError:
        print(
            "Нужен пакет для DuckDuckGo-поиска: pip install ddgs",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    _require_ddgs()
    parser = argparse.ArgumentParser(description="Профиль → три Markdown-отчёта (мультиагент).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "input.txt",
        help="Файл с блоками Компетенции / Интересы",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Каталог для .md файлов",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="model_id для InferenceClientModel",
    )
    args = parser.parse_args()

    if not get_token():
        print(
            "Предупреждение: не найден HF-токен (HF_TOKEN или huggingface-cli login). "
            "Запросы к Inference API могут не пройти.\n",
            file=sys.stderr,
        )

    if not args.input.is_file():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        sys.exit(1)

    profile = parse_input_txt(args.input)
    print(f"Профиль загружен из {args.input}", flush=True)
    run_pipeline(profile, args.out.resolve(), args.model)
    print("Готово.", flush=True)


if __name__ == "__main__":
    main()

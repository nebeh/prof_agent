"""Пример CodeAgent (smolagents 1.24+) с поиском DuckDuckGo и HF Inference.

Запуск в окружении micromamba:
    micromamba activate deep
    export HF_TOKEN=...   # токен с правом Inference Providers: https://hf.co/settings/tokens
    python agent.py
"""

from __future__ import annotations

import sys

from huggingface_hub import get_token
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel


def main() -> None:
    if not get_token():
        print(
            "Нет Hugging Face токена: задайте HF_TOKEN или выполните `huggingface-cli login`.\n"
            "Для Inference Providers токену нужны соответствующие права: https://hf.co/settings/tokens",
            file=sys.stderr,
        )

    # HuggingFaceInferenceServerModel переименован в InferenceClientModel (smolagents ≥1.x)
    model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    search_tool = DuckDuckGoSearchTool()
    agent = CodeAgent(
        tools=[search_tool],
        model=model,
        add_base_tools=True,
    )

    print("--- Запуск агента ---")
    prompt = """
Найди актуальные тренды в области Python-разработки и системного анализа в России на 2026 год.
На основе найденного предложи 2 бесплатных и 1 платный ресурс для обучения в РФ.
Составь краткий план обучения на первую неделю.
"""
    agent.run(prompt)


if __name__ == "__main__":
    main()

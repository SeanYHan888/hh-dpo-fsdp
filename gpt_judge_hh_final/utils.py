"""Shared utilities for HH dataset processing and chat templates."""

import re
from typing import Dict, List, Optional, Any
from datasets import load_dataset, Dataset


TAG_RE = re.compile(r"\n\n(Human|Assistant): ?")
HUMAN_TAG = "\n\nHuman:"
ASSISTANT_TAG = "\n\nAssistant:"

# Llama 3 chat template
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = message['content'] %}"
    "{% if loop.index0 == 0 %}"
    "{{ '<|begin_of_text|>' }}"
    "{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + content | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def strip_one_leading_newline(text: str) -> str:
    """Remove a single leading newline to normalize HH blocks."""
    return text[1:] if text.startswith("\n") else text


def parse_hh_to_messages(text: str) -> List[Dict[str, str]]:
    """Parse Anthropic HH multi-turn text into [{role, content}, ...].

    Ensures content is trimmed and skips empty blocks.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("\n\nHuman:") and not text.startswith("\n\nAssistant:"):
        text = "\n\n" + text

    parts = TAG_RE.split(text)
    messages = []
    for i in range(1, len(parts), 2):
        role_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        content = strip_one_leading_newline(content).strip()
        if not content:
            continue
        role = "user" if role_tag == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def split_prompt_and_response(input_text: str) -> tuple:
    """Split HH format text into prompt and response.

    HH format: multi-turn text containing many "\\n\\nAssistant:".
    We take the LAST Assistant tag as the start of the final assistant response.
    """
    input_text = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = input_text.rfind(ASSISTANT_TAG)
    if index < 0:
        raise ValueError("No '\\n\\nAssistant:' tag found in HH input.")
    prompt = input_text[: index + len(ASSISTANT_TAG)]
    response = input_text[index + len(ASSISTANT_TAG) :]
    response = strip_one_leading_newline(response)
    return prompt, response


def convert_to_triples(
    chosen_text: str, rejected_text: str
) -> Optional[Dict[str, str]]:
    """Convert one HH row into an explicit triplet: {prompt, chosen, rejected}."""
    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)

    if not rejected_text.startswith(chosen_prompt):
        return None

    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt) :])

    if len(chosen_prompt.strip()) == 0:
        return None
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0:
        return None

    return {
        "prompt": chosen_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def build_hh_dataset(ds) -> Dataset:
    """Process entire dataset into HH triplets format."""
    hh_ds_raw = []
    for idx, row in enumerate(ds):
        output = convert_to_triples(
            chosen_text=row["chosen"], rejected_text=row["rejected"]
        )
        if output is not None:
            hh_ds_raw.append(output)
    return Dataset.from_list(hh_ds_raw)


def extract_single_turn_instruction(text: str) -> Optional[str]:
    """Extract instruction from single-turn HH conversations only."""
    messages = parse_hh_to_messages(text)
    if len(messages) != 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    return messages[0]["content"]


def extract_instruction_from_prompt(prompt_text: str) -> Optional[str]:
    """Extract user instruction from HH prompt format (ending with Assistant:)."""
    if not isinstance(prompt_text, str):
        return None
    if not prompt_text.endswith(ASSISTANT_TAG):
        return None
    # Single-turn guard: exactly one Human tag and one Assistant tag
    if prompt_text.count(HUMAN_TAG) != 1 or prompt_text.count(ASSISTANT_TAG) != 1:
        return None
    start = prompt_text.find(HUMAN_TAG)
    if start < 0:
        return None
    start += len(HUMAN_TAG)
    end = prompt_text.find(ASSISTANT_TAG, start)
    if end < 0:
        return None
    return prompt_text[start:end].strip()


def format_prompt(tokenizer: Any, instruction: str, apply_chat_template: bool) -> str:
    """Format instruction as prompt for model generation."""
    if not apply_chat_template:
        return f"\n\nHuman: {instruction}\n\nAssistant:"
    if (
        apply_chat_template
        and getattr(tokenizer, "apply_chat_template", None)
        and tokenizer.chat_template
    ):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    bos_token = tokenizer.bos_token or ""
    return f"{bos_token}{instruction}"


def load_hh_single_turn_instructions(
    repo_id: str = "Anthropic/hh-rlhf",
    data_dir: str = "harmless-base",
    split: str = "test",
) -> List[str]:
    """Load HH dataset and extract single-turn instructions only."""
    dataset = load_dataset(repo_id, data_dir=data_dir, split=split)
    instructions = []
    for row in dataset:
        text = row.get("chosen") or row.get("prompt") or row.get("text")
        if text is None:
            continue
        instruction = extract_single_turn_instruction(text)
        if instruction is None:
            continue
        instructions.append(instruction)
    return instructions

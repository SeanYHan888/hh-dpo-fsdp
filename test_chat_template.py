#!/usr/bin/env python
"""
Test script for chat template conversion.
Run this to verify the HH -> Llama 3 format conversion works correctly.

Usage: python test_chat_template.py
"""

from process_hh_dataset import parse_hh_to_messages, apply_chat_template_to_triplet


def test_parse_hh_to_messages():
    """Test parsing HH format into message list."""
    print("=" * 60)
    print("Test 1: parse_hh_to_messages()")
    print("=" * 60)

    # Single turn
    hh_single = "\n\nHuman: Hello there\n\nAssistant: Hi! How can I help?"
    messages = parse_hh_to_messages(hh_single)
    print(f"Input: {repr(hh_single)}")
    print(f"Output: {messages}")
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello there"}
    assert messages[1] == {"role": "assistant", "content": "Hi! How can I help?"}
    print("✓ Single turn passed\n")

    # Multi turn
    hh_multi = (
        "\n\nHuman: Hello\n\nAssistant: Hi!\n\nHuman: What is 2+2?\n\nAssistant: 4"
    )
    messages = parse_hh_to_messages(hh_multi)
    print(f"Input: {repr(hh_multi)}")
    print(f"Output: {messages}")
    assert len(messages) == 4
    assert messages[2] == {"role": "user", "content": "What is 2+2?"}
    assert messages[3] == {"role": "assistant", "content": "4"}
    print("✓ Multi turn passed\n")

    # Prompt only (ends with Assistant tag, no response)
    hh_prompt = "\n\nHuman: Hello\n\nAssistant:"
    messages = parse_hh_to_messages(hh_prompt)
    print(f"Input: {repr(hh_prompt)}")
    print(f"Output: {messages}")
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "Hello"}
    print("✓ Prompt-only passed\n")

    print("All parse_hh_to_messages tests passed!\n")


def test_apply_chat_template():
    """Test chat template application (requires transformers)."""
    print("=" * 60)
    print("Test 2: apply_chat_template_to_triplet()")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("⚠ transformers not installed, skipping this test")
        return

    # Use a small Llama tokenizer for testing
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except Exception as e:
        print(f"⚠ Could not load Llama tokenizer: {e}")
        print("  Trying fallback tokenizer...")
        try:
            tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
        except:
            print("⚠ Could not load any tokenizer, skipping test")
            return

    triplet = {
        "prompt": "\n\nHuman: Hello\n\nAssistant:",
        "chosen": "Hi there! How can I help you today?",
        "rejected": "Go away.",
    }

    print(f"Input triplet:")
    print(f"  prompt: {repr(triplet['prompt'])}")
    print(f"  chosen: {repr(triplet['chosen'])}")
    print(f"  rejected: {repr(triplet['rejected'])}")

    formatted = apply_chat_template_to_triplet(triplet, tok)

    print(f"\nOutput triplet:")
    print(f"  prompt: {repr(formatted['prompt'][:100])}...")
    print(f"  chosen: {repr(formatted['chosen'])}")
    print(f"  rejected: {repr(formatted['rejected'])}")

    # Basic assertions
    assert (
        "user" in formatted["prompt"].lower()
        or "<|start_header_id|>" in formatted["prompt"]
    )
    assert len(formatted["chosen"]) > 0
    assert len(formatted["rejected"]) > 0

    print("\n✓ apply_chat_template_to_triplet passed!\n")


if __name__ == "__main__":
    test_parse_hh_to_messages()
    test_apply_chat_template()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)

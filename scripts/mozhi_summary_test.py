from __future__ import annotations

import argparse

from nava_core.mozhi.chat.client import ChatClient
from nava_core.shared.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Mozhi summary model call")
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "USER: My rice leaves have brown spots and yellowing.\n"
            "ASSISTANT: Check for rice blast and consider approved fungicides.\n"
            "USER: What should I spray now?\n"
            "ASSISTANT: Use local guidelines for dosage; consult agronomist.\n"
            "USER: Write a simple C program.\n"
            "ASSISTANT: Sorry, I can't assist with that.\n"
            "USER: What did we discuss?\n"
            "ASSISTANT: We discussed your rice leaves having brown spots and yellowing, potential rice blast disease, and the importance of following local guidelines for fungicide use. You also asked for a C program, which is out of scope. So I refused to provide one."
        ),
        help="Conversation snippet to summarize",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = ChatClient.from_settings()

    messages = [
        {
            "role": "system",
            "content": (
                "Create a chat memory that preserves who asked what and how NAVA replied. "
                "Output 4-8 bullet points only. No headings or preamble. "
                "Each bullet must include both parts in this format: "
                "'User: ... | NAVA: ...'. "
                "Read the full assistant reply before summarizing; capture all key recommendations, "
                "steps, and cautions, not just the opening lines. "
                "Merge related details into one bullet; keep concrete numbers, timing, and constraints. "
                "Focus on agricultural content: crops, symptoms, suspected issues, actions, advice, "
                "constraints, locations, timelines, and open questions. "
                "Exclude any out-of-scope requests and refusals. "
                "Keep each bullet concise and specific."
            ),
        },
        {"role": "user", "content": args.text},
    ]

    reply, error = client.send(
        messages,
        model_override=settings.hf_summary_model,
        temperature_override=settings.hf_summary_temperature,
        max_new_tokens_override=settings.hf_summary_max_new_tokens,
    )

    if error:
        print(f"Error: {error}")
        return 1

    print(f"Summary model: {settings.hf_summary_model}\n")
    print("Summary model reply:\n")
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

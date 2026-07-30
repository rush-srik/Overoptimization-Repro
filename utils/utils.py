def apply_proxy_chat_template(entry):
    """
    Pythia doesn't have a chat template, so this applies a custom one
    directly to the labeled gold data.
    """
    return {
        "prompt": f"Human: {entry['prompt']}\n\nAssistant:",
        "chosen": " " + entry["chosen"].strip(),
        "rejected": " " + entry["rejected"].strip()
    }
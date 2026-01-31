import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial

ASSISTANT_TAG = "\n\nAssistant:"

# delete the \n at the beginning of the response
def strip_one_leading_newline(s): 
    return s[1:] if s.startswith("\n") else s

def split_prompt_and_response(input_text):
    """
    HH format: multi-turn text containing many "\n\nAssistant:".
    We take the LAST Assistant tag as the start of the final assistant response.

    Returns:
    prompt: everything up to and including the final "\n\nAssistant:"
    response: the assistant completion after that tag (no leading newline)
    
    """
    input_text = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = input_text.rfind(ASSISTANT_TAG)
    if index < 0:
        raise ValueError("No '\\n\\nAssistant:' tag found in HH input.")
    prompt = input_text[:index + len(ASSISTANT_TAG)]
    response = input_text[index + len(ASSISTANT_TAG):]
    response = strip_one_leading_newline(response)
    return prompt, response


def convert_to_triples(chosen_text, rejected_text):
    """
    convert one HH row into an explicit triplet:
      {prompt, chosen, rejected}

    """
    # get prompt and response from chosen_text
    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)

    # assume the chosen and rejected prompts are same
    if not rejected_text.startswith(chosen_prompt):
        return None
    
    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt):])
    
    
    if len(chosen_prompt.strip()) == 0:
        return None
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0:
        return None
    
    return {"prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response}

# process entire dataset, build hh dataset
def build_HH_dataset(ds):
    hh_ds_raw = []
    for idx, row in enumerate(ds):
        output = convert_to_triples(
            chosen_text = row['chosen'],
            rejected_text = row['rejected']
        )
        if output is not None:
            hh_ds_raw.append(output)
    return Dataset.from_list(hh_ds_raw)


def collate_fn(batch, tokenizer, max_len):
    """
    Input: list of triplets:
      {prompt, chosen, rejected}

    Output tensors:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels
      prompt_length

    """

    chosen_txt, rejected_txt = [], []
    prompt_lens = []

    if tokenizer.pad_token_id is None:
        # decoder-only models often have no pad token; reuse eos as pad for batching
        tokenizer.pad_token = tokenizer.eos_token

    for item in batch:
        prompt = str(item['prompt'])
        chosen = str(item['chosen'])
        rejected = str(item['rejected'])

        chosen_txt.append(prompt + chosen)
        rejected_txt.append(prompt + rejected)

        # prompt_length without special tokens
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_lens.append(len(prompt_ids))

    # chosen, rejected tokens
    enc_chosen = tokenizer(
        chosen_txt,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    enc_rejected = tokenizer(
        rejected_txt,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # If tokenizer adds a BOS token at position 0, prompt length should include it.
    has_bos = (
        tokenizer.bos_token_id is not None
        and enc_chosen.input_ids.size(1) > 0
        and enc_chosen.input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    bos_shift = 1 if has_bos else 0
    prompt_length = torch.tensor([pl + bos_shift for pl in prompt_lens], dtype=torch.long)

    # Build labels: input_ids with prompt tokens masked to -100
    chosen_labels = enc_chosen.input_ids.clone()
    rejected_labels = enc_rejected.input_ids.clone()

    max_seq_len = chosen_labels.size(1)
    for i, pl in enumerate(prompt_length.tolist()):
        pl = min(pl, max_seq_len)
        chosen_labels[i, :pl] = -100
        rejected_labels[i, :pl] = -100

    # mask padding
    chosen_labels[enc_chosen.attention_mask == 0] = -100
    rejected_labels[enc_rejected.attention_mask == 0] = -100

    return {
        "chosen_input_ids": enc_chosen.input_ids,
        "chosen_attention_mask": enc_chosen.attention_mask,
        "chosen_labels": chosen_labels,

        "rejected_input_ids": enc_rejected.input_ids,
        "rejected_attention_mask": enc_rejected.attention_mask,
        "rejected_labels": rejected_labels,

        # optional: useful for debugging / analysis
        "prompt_length": prompt_length,
    }

# build distributed train, eval dataloader
def build_train_val_dist(config, tokenizer, rank, world_size):
    ds_cfg = config['dataset']
    raw_dataset = ds_cfg['dataset_name']
    split = ds_cfg['subset']
    val_ratio = float(ds_cfg['val_ratio'])
    seed = int(ds_cfg['seed'])
    max_len = int(ds_cfg['max_len'])
    batch_size = int(config['dpo_training']['batch_size'])

    ds = load_dataset(
        raw_dataset, 
        data_dir="harmless-base",
        split=split
        )
    ds_triple = build_HH_dataset(ds)

    split_ds = ds_triple.train_test_split(test_size=val_ratio, seed=seed)
    train_ds_raw, val_ds_raw = split_ds["train"], split_ds["test"]

    ds_collate = partial(collate_fn, tokenizer=tokenizer, max_len=max_len)

    train_sampler = DistributedSampler(
        train_ds_raw, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = DistributedSampler(
        val_ds_raw, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        train_ds_raw,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=ds_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=ds_collate,
        pin_memory=True,
    )

    return train_loader, val_loader
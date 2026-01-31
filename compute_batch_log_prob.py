import torch
from dpo_loss import compute_log_prob

def compute_batch_log_prob(batch, policy, ref_model):
    # concatenate chosen and rejected along the batch dimension
    chosen_ids = batch['chosen_input_ids']
    rejected_ids = batch['rejected_input_ids']

    batch_size = chosen_ids.size(0)

    input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
    attention_mask = torch.cat(
        [batch['chosen_attention_mask'
         batch['rejected_attention_mask']], 
         dim=0)
    labels = torch.cat(
        [batch['chosen_labels'], 
         batch['rejected_labels']], 
         dim=0)
    
    # policy forward
    policy_logits = policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    policy_log_prob = compute_log_prob(
        logits=policy_logits,
        labels=labels,
    )
    policy_chosen_log_prob, policy_rejected_log_prob =  policy_log_prob.split(batch_size, dim=0)

    # ref forward (no grad)
    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        ref_log_prob = compute_log_prob(
            logits=ref_logits,
            labels=labels,
        )
        ref_chosen_log_prob, ref_rejected_log_prob = ref_log_prob.split(batch_size, dim=0)
    
    return policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob
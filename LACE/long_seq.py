import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens, 
                       use_entity_embedding_layers_mean, use_entity_attention_layers_mean):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)

    output_hidden_states = use_entity_embedding_layers_mean

    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=output_hidden_states
        )

        if use_entity_embedding_layers_mean:
            sequence_output = torch.stack(output.hidden_states, dim=0)
        else:
            sequence_output = output[0]

        if use_entity_attention_layers_mean:
            attention = torch.stack(output.attentions, dim=0)  # Toutes les couches d'attention
        else:
            attention = output[-1][-1]  # Dernière couche d'attention
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()

        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=output_hidden_states
        )

        if use_entity_embedding_layers_mean:
            sequence_output = torch.stack(output.hidden_states, dim=0)
        else:
            sequence_output = output[0]

        if use_entity_attention_layers_mean:
            attention = torch.stack(output.attentions, dim=0)
        else:
            attention = output[-1][-1]

        i = 0
        new_output, new_attention = [], []

        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                if use_entity_embedding_layers_mean:
                    output = F.pad(sequence_output[:, i], (0, 0, 0, c - 512))
                else:
                    output = F.pad(sequence_output[i], (0, 0, 0, c - 512))

                if use_entity_attention_layers_mean:
                    att = F.pad(attention[:, i], (0, c - 512, 0, c - 512))
                else:
                    att = F.pad(attention[i], (0, c - 512, 0, c - 512))

                new_output.append(output)
                new_attention.append(att)

            elif n_s == 2:
                if use_entity_embedding_layers_mean:
                    output1 = sequence_output[:, i, :512 - len_end]
                    output2 = sequence_output[:, i + 1, len_start:]
                    output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                    output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                else:
                    output1 = sequence_output[i][:512 - len_end]
                    output2 = sequence_output[i + 1][len_start:]
                    output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                    output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))

                if use_entity_attention_layers_mean:
                    att1 = attention[:, i, :, :512 - len_end, :512 - len_end]
                    att2 = attention[:, i + 1, :, len_start:, len_start:]
                    att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))
                    att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                else:
                    att1 = attention[i][:, :512 - len_end, :512 - len_end]
                    att2 = attention[i + 1][:, len_start:, len_start:]
                    att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))
                    att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])

                mask1 = attention_mask[i][:512 - len_end]
                mask2 = attention_mask[i + 1][len_start:]

                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))

                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2) / (att1 + att2).sum(-1, keepdim=True).clamp(min=1e-10)

                new_output.append(output)
                new_attention.append(att)

            i += n_s

        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)

    return sequence_output, attention

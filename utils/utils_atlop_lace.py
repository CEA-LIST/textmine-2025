import pandas as pd
import torch
import random
from apex import amp
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from copy import deepcopy
import math

from utils.utils import calculate_f1_scores_for_dataframe, check_if_relation_is_possible, get_potential_relations
from utils.ontology import TYPE_TO_PARENT

def load_data(file):
    df = pd.read_json(file)
    df = df.set_index('title', drop=False)
    df.index.name = 'id'
    return df

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Forcing deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    # print(max_len)
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    labels = [f["labels"] for f in batch if "labels" in f]
    sent_pos =  [f["sent_pos"] for f in batch if 'sent_pos' in f]
    sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
    teacher_logits = [f["teacher_logits"] for f in batch if "teacher_logits" in f]
    negative_mask = [f["negative_mask"] for f in batch if "negative_mask" in f]

    # Convertir les input_ids et input_mask en tenseurs
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    
    if len(labels) == 0:
        labels = None
    if len(sent_pos) == 0:
        sent_pos = None
    if len(teacher_logits) == 0:
        teacher_logits = None
    if len(negative_mask) == 0:
        negative_mask = None
    else:
        negative_mask = torch.stack(negative_mask, dim=0)

    if sent_labels != [] and None not in sent_labels:
        sent_labels_tensor = []
        max_sent = max([len(f["sent_pos"]) for f in batch])
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None
    
    output = (input_ids, input_mask, entity_pos, hts, labels, sent_pos, sent_labels_tensor, teacher_logits, negative_mask)
    
    return output

def print_statistics(df):
    # Number of documents
    num_documents = len(df)

    # Count positive examples
    # positive examples are those where the first element is not 1 (UNRELATED)
    pos_samples = df['labels'].apply(lambda x: sum(np.array(x)[0] != 1)).sum()

    # Count total possible pairs of entities
    total_entities = len(df['entities'].iloc[0])
    total_possible_pairs = num_documents * (total_entities ** 2)

    # Calculate negative examples (everything else is negative, i.e., labeled as UNRELATED)
    neg_samples = total_possible_pairs - pos_samples

    # Print statistics
    print("# of documents in training set: {}".format(num_documents))
    print("# of positive examples in training set: {}".format(pos_samples))
    print("# of negative examples in training set: {}".format(neg_samples))

def create_entity_types_to_tokens_mapping(df, use_specific_entity_types_markers, use_specific_parent_entity_types_markers):
    entity_types_to_tokens_mapping = {}
    tokens_to_add_to_vocab = []
    for i, row in df.iterrows():
        entities = row['entities']
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in list(entity_types_to_tokens_mapping.keys()):
                if use_specific_entity_types_markers:
                    entity_types_to_tokens_mapping[entity_type] = {"boe_token": f'<{entity_type}>', "eoe_token": f'</{entity_type}>'}
                    tokens_to_add_to_vocab.append(f'<{entity_type}>')
                    tokens_to_add_to_vocab.append(f'</{entity_type}>')
                elif use_specific_parent_entity_types_markers:
                    entity_type = TYPE_TO_PARENT.get(entity_type)
                    entity_types_to_tokens_mapping[entity_type] = {"boe_token": f'<{entity_type}>', "eoe_token": f'</{entity_type}>'}
                    tokens_to_add_to_vocab.append(f'<{entity_type}>')
                    tokens_to_add_to_vocab.append(f'</{entity_type}>')
                else:
                    entity_types_to_tokens_mapping[entity_type] = {"boe_token": "*", "eoe_token": "*"}

    return entity_types_to_tokens_mapping, tokens_to_add_to_vocab


def add_entity_markers(sample, tokenizer, entity_start, entity_end, entity_types, entity_types_to_tokens_mapping, use_specific_parent_entity_types_markers):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
    # add * marks to the beginning and end of entities
        new_map = {}
        
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                entity_type = entity_types[(i_s, i_t)]
                if use_specific_parent_entity_types_markers:
                    entity_type = TYPE_TO_PARENT.get(entity_type)
                tokens_wordpiece = [entity_types_to_tokens_mapping[entity_type]['boe_token']] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                entity_type = entity_types[(i_s, i_t)]
                if use_specific_parent_entity_types_markers:
                    entity_type = TYPE_TO_PARENT.get(entity_type)
                tokens_wordpiece = tokens_wordpiece + [entity_types_to_tokens_mapping[entity_type]['eoe_token']]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        
        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end
        
        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)
    return sents, sent_map, sent_pos

def get_sent_pos(sample):
    ''' Return the start and end positions of each sentence in the sample. '''

    sent_pos = []
    sent_start = 0

    for sent in sample['sents']:
        sent_length = len(sent)
        sent_end = sent_start + sent_length
        sent_pos.append((sent_start, sent_end))
        sent_start = sent_end

    return sent_pos

def format_data(example, tokenizer, rel2id, entity_types_to_tokens_mapping, use_specific_parent_entity_types_markers, max_seq_length=1024, eval=False, get_silver_evidence_train=False, get_silver_evidence_test=False):
    sents = []
    sent_map = []
    entities = example['vertexSet']
    entity_start, entity_end = [], []
    entity_types = {}
    
    for entity in entities:
        for mention in entity:
            sent_id = mention["sent_id"]
            pos = mention["pos"]
            entity_start.append((sent_id, pos[0],))
            entity_end.append((sent_id, pos[1] - 1,))
            entity_type = mention['type']
            entity_types[(sent_id, pos[0])] = entity_type
            entity_types[(sent_id, pos[1]-1)] = entity_type

    sents, sent_map, sent_pos = add_entity_markers(example, tokenizer, entity_start, entity_end, entity_types, entity_types_to_tokens_mapping, use_specific_parent_entity_types_markers)

    train_triple = {}
    if "labels_for_atlop" in example:
        for label in example['labels_for_atlop']:
            evidence = label['evidence']
            r = int(rel2id[label['r']])
            if (label['h'], label['t']) not in train_triple:
                train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
            else:
                train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})

    entity_pos = []
    for e in entities:
        entity_pos.append([])
        for m in e:
            start = sent_map[m["sent_id"]][m["pos"][0]]
            end = sent_map[m["sent_id"]][m["pos"][1]]
            entity_pos[-1].append((start, end,))
    
    relations, hts = [], []
    
    for h, t in train_triple.keys():
        relation = [0] * len(rel2id)
        for mention in train_triple[h, t]:
            relation[mention["relation"]] = 1
        if [h, t] not in hts:
            relations.append(relation)
            hts.append([h, t])

    if not get_silver_evidence_train:
        for h in range(len(entities)):
            for t in range(len(entities)):
                entity_1 = next((entity for entity in example['entities'] if entity['id'] == h), None)
                entity_2 = next((entity for entity in example['entities'] if entity['id'] == t), None)
                if [h, t] not in hts and (not eval or (eval and check_if_relation_is_possible(entity_1, entity_2))):
                    relation = [1] + [0] * (len(rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])

    sents = sents[:max_seq_length - 2] # truncate, -2 for [CLS] and [SEP]
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    # print(len(input_ids))
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    # print(len(input_ids))
    # print('###########')

    # Add new features to the DataFrame example
    example['input_ids'] = input_ids
    example['entity_pos'] = entity_pos
    example['labels'] = relations
    example['hts'] = hts
    if "labels_for_atlop" in example:
        example['labels'] = relations
        # Ensure the relations count matches
        if not eval and not get_silver_evidence_train and not get_silver_evidence_test:
            assert len(relations) == len(entities) * len(entities)
    if get_silver_evidence_train or get_silver_evidence_test:
        example['sent_pos'] = sent_pos
    return example

def filter_preds(pred_rels, pred_rel, example, idx_h, idx_t):
    # Add predicted relations
    entity_1 = next((entity for entity in example['entities'] if entity['id'] == idx_h), None)
    entity_2 = next((entity for entity in example['entities'] if entity['id'] == idx_t), None)
    if pred_rel in get_potential_relations(entity_1, entity_2):
        # Only add GENDER_MALE or GENDER_FEMALE if idx_h == idx_t
        if pred_rel in ['GENDER_MALE', 'GENDER_FEMALE'] and idx_h == idx_t:
            pred_rels.append([idx_h, pred_rel, idx_t])
        elif pred_rel not in ['GENDER_MALE', 'GENDER_FEMALE']:
            pred_rels.append([idx_h, pred_rel, idx_t])
    return pred_rels

def get_preds(example, args, model, id2rel, dreeam_model=False, eider_model=False):

    batch = collate_fn([example])

    inputs = {'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'entity_pos': batch[2],
            'hts': batch[3],
            }
    
    if dreeam_model or eider_model:
        inputs['sent_pos'] = batch[5]

    model.eval()
    preds = []
    with torch.no_grad():
        # Get model predictions
        if dreeam_model:
            pred = model(**inputs, tag='test')
            pred = pred['rel_pred']
        else:
            pred, *_ = model(**inputs)
        pred = pred.cpu().numpy()
        pred[np.isnan(pred)] = 0
        preds.append(pred)
    
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    pred_rels = []
    unfiltered_preds = []
    for i, (idx_h, idx_t) in enumerate(example['hts']):
        pred = preds[i]
        max_rel_idx = np.argmax(pred)
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if args.multi_label: 
                if p != 0:
                    pred_rel = id2rel[p]  # Convert index to relation name
                    if pred_rel != 'UNRELATED':
                        unfiltered_preds.append([idx_h, pred_rel, idx_t])
                        pred_rels = filter_preds(pred_rels, pred_rel, example, idx_h, idx_t)
            else:
                if p == max_rel_idx:
                    pred_rel = id2rel[p] 
                    if pred_rel != 'UNRELATED':
                        unfiltered_preds.append([idx_h, pred_rel, idx_t])
                        pred_rels = filter_preds(pred_rels, pred_rel, example, idx_h, idx_t)

                
    example['unfiltered_predictions'] = unfiltered_preds
    example['formatted_predictions'] = pred_rels  # Add predictions to the row
    return example

def create_id2rel(rel2id):
    return {v: k for k, v in rel2id.items()}

def worker_init_fn(worker_id, args):
    np.random.seed(args.seed + worker_id)
    random.seed(args.seed + worker_id)

def extract_features_from_df(df):
    features = []
    for _, row in df.iterrows():
        feature = {
            'input_ids': row['input_ids'],
            'entity_pos': row['entity_pos'],
            'labels': row['labels'],
            'hts': row['hts'],
        }
        
        if 'sent_pos' in row:
                feature['sent_pos'] = row['sent_pos']
        if 'sent_labels' in row:
                feature['sent_labels'] = row['sent_labels']
        features.append(feature)
    return features

def get_random_mask(train_features, drop_prob):
    new_features = []  
    n_e = 60
    for old_feature in train_features:
        feature = deepcopy(old_feature)
        neg_labels = torch.tensor(feature['labels'])[:, 0]
        neg_index = torch.where(neg_labels==1)[0]
        pos_index = torch.where(neg_labels==0)[0]
        perm = torch.randperm(neg_index.size(0))
        sampled_negative_index = neg_index[perm[:int(drop_prob * len(neg_index))]]
        neg_mask = torch.ones(len(feature['labels']))
        neg_mask[sampled_negative_index] = 0
        #feature['negative_mask'] = neg_mask        
        pad_neg = torch.zeros((n_e, n_e))
        num_e = int(math.sqrt(len(neg_mask)))
        pad_neg[:num_e,:num_e] = neg_mask.view(num_e,num_e)
        feature['negative_mask'] = pad_neg
        new_features.append(feature)
    return new_features

def train(args, model, df_train, df_dev, id2rel, lace_model=False, dreeam_model=False, eider_model=False, kd_docre_model=False, fold=None):
    def finetune(features, optimizer, num_epoch, num_steps, id2rel, args):
        best_f1_score = -1
        best_model_step = None
        loss_values = []  # Liste pour stocker les pertes
        if kd_docre_model:
            features = get_random_mask(features, args.drop_prob)
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, 
                                      collate_fn=collate_fn, drop_last=True, 
                                      worker_init_fn=lambda worker_id: worker_init_fn(worker_id, args))
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))

        if dreeam_model:
            scaler = GradScaler() 

        for epoch in tqdm(train_iterator, desc='Train epoch'):
            print(f"\n{'*' * 20} Epoch {epoch + 1} {'*' * 20}\n")
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {
                    'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'labels': batch[4],
                    'entity_pos': batch[2],
                    'hts': batch[3],
                }

                if dreeam_model:
                    inputs['labels'] = [torch.tensor(label) for label in inputs['labels']] 
                    inputs['labels']  = torch.cat(inputs['labels'] , dim=0).to(args.device)

                    inputs['sent_pos'] = batch[5]
                    inputs['sent_labels'] = batch[6]
                elif eider_model:
                    inputs['sent_pos'] = batch[5]
                    inputs['sent_labels'] = batch[6]
                elif kd_docre_model:
                    inputs['negative_mask'] = batch[8].to(args.device)

                outputs = model(**inputs)

                # Gestion des loss
                if dreeam_model:
                    loss = [outputs["loss"]["rel_loss"][0]]
                    if inputs.get("sent_labels", None) is not None:
                        loss.append(outputs["loss"]["evi_loss"] * int(args.evi_lambda))
                    loss = sum(loss) / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()  # Backward pass avec GradScaler pour précision mixte
                else:
                    if kd_docre_model:
                        loss = outputs / args.gradient_accumulation_steps
                    else:
                        loss = outputs[0] / args.gradient_accumulation_steps
                    if args.device == torch.device("cuda") and not lace_model and args.transformer_type not in ['deberta-v2']:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                # Stocker la loss dans la liste
                loss_values.append(loss.item())

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        if dreeam_model:
                            scaler.unscale_(optimizer)  # Unscale les gradients avant le clipping dans dreeam_model
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        else:
                            if args.device == torch.device("cuda") and not lace_model and args.transformer_type not in ['deberta-v2']:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if dreeam_model:
                        scaler.step(optimizer)  # Met à jour les poids avec le scaler pour dreeam_model
                        scaler.update()  # Met à jour le scaler
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if num_steps % 100 == 0:
                    print(f"Step {num_steps}: Loss = {loss.item():.4f}")

                if args.train_on_full_train_set == False and args.cross_validation == False:
                    # Évaluation périodique
                    if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                        if dreeam_model:
                            dev_results, _, _ = evaluate(args, model, df_dev, id2rel, dreeam_model=True) 
                        if eider_model:
                            dev_results, _, _ = evaluate(args, model, df_dev, id2rel, eider_model=True) 
                        else:
                            dev_results, _, _ = evaluate(args, model, df_dev, id2rel) 
                        print(f"Development set results:")
                        print(f"  Micro F1 Score: {dev_results['micro_scores']['f_score']:.4f}%")
                        print(f"  Macro F1 Score: {dev_results['macro_scores']['f_score']:.4f}%")

                        if dev_results['macro_scores']['f_score'] > best_f1_score:
                            best_f1_score = dev_results['macro_scores']['f_score']
                            best_model_step = num_steps  # Stocker l'étape du meilleur modèle
                            if args.save_finetuned_model_path != "":
                                os.makedirs(os.path.dirname(args.save_finetuned_model_path), exist_ok=True)
                                torch.save(model.state_dict(), args.save_finetuned_model_path)
                                print(f"Best model saved with macro F1 score: {best_f1_score:.4f}%")
                elif (args.train_on_full_train_set or args.cross_validation)and (epoch + 1 == num_epoch) and (step + 1 == len(train_dataloader)):
                    # Save model after final epoch and last step
                    if args.save_finetuned_model_path != "":
                        os.makedirs(os.path.dirname(args.save_finetuned_model_path), exist_ok=True)
                        torch.save(model.state_dict(), args.save_finetuned_model_path)
                        print(f"Model saved after full training on the dataset.")


        # Tracer la courbe de la loss après toutes les époques
        save_loss_curve(loss_values, args.pred_path, best_model_step, fold)

        return num_steps

    train_features = extract_features_from_df(df_train)

    new_layer = ["extractor", "bilinear"]
    if eider_model:
        new_layer.extend(['sr_bilinear'])
    if kd_docre_model:
        new_layer.extend(['classifier', 'projection'])

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": float(args.lr_added)},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate), eps=float(args.adam_epsilon))

    if args.device == torch.device("cuda") and not lace_model and not dreeam_model and args.transformer_type not in ['deberta-v2']:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    else:
        model.to(args.device)  # Toujours s'assurer que le modèle est sur l'appareil

    num_steps = 0
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps, id2rel, args)


def save_loss_curve(loss_values, pred_path, best_model_step, fold=None):
    """ Fonction pour sauvegarder la courbe de la loss. """
    plt.figure(figsize=(10,6))
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # Ajouter une ligne verticale rouge à l'étape du meilleur modèle
    if best_model_step is not None:
        plt.axvline(x=best_model_step, color='red', linestyle='--', label=f'Model saved, step: {best_model_step}')
        plt.legend()

    # Extraire le dossier et modifier l'extension du fichier
    if fold is None:
        save_path = os.path.dirname(pred_path) + "/loss.png"
    else:
        save_path = os.path.dirname(pred_path) + f"/loss_fold={fold}.png"

    plt.savefig(save_path)
    print(f"Loss curve saved at: {save_path}")



def evaluate(args, model, df, id2rel, dreeam_model=False, eider_model=False):
    # Use the get_preds function to get predictions for the DataFrame
    if dreeam_model:
        df = df.apply(lambda row: get_preds(row, args, model, id2rel, dreeam_model=True), axis=1)
    elif eider_model:
        df = df.apply(lambda row: get_preds(row, args, model, id2rel, eider_model=True), axis=1)
    else:
        df = df.apply(lambda row: get_preds(row, args, model, id2rel), axis=1)

    # Apply the provided functions for F1 score calculation and result formatting
    results = calculate_f1_scores_for_dataframe(df, 'formatted_predictions')
    results_unfiltered = calculate_f1_scores_for_dataframe(df, 'unfiltered_predictions')

    return results, results_unfiltered, df
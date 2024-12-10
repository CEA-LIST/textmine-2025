import torch
import numpy as np
import pickle

def get_num_entities(df):
    entity_ids = []  
    for _, row in df.iterrows():
        entities = row['entities']
        for entity in entities:
            entity_ids.append(entity)  
    return len(entity_ids)


def create_adj_matrix(df, adjacency_matrix_save_path):
    num_entities = get_num_entities(df)  
    adj_matrix = np.zeros((num_entities, num_entities))

    for _, row in df.iterrows():
        relations = row['relations']
        for rel in relations:
            subj_id, _, obj_id = rel
            adj_matrix[subj_id, obj_id] = 1
            adj_matrix[obj_id, subj_id] = 1  # Assuming bidirectional relation
    
    # Normalization and thresholding (like in the code's gen_dgl_graph)
    nums = np.sum(adj_matrix, axis=0)
    _nums = nums[:, np.newaxis]
    for i in range(len(_nums)):
        if _nums[i] > 10:
            adj_matrix[i] = adj_matrix[i] / _nums[i]
        else:
            adj_matrix[i] = 0

    # Applying threshold t=0.05 and probability p=0.3
    t, p = 0.05, 0.3
    adj_matrix[adj_matrix < t] = 0
    adj_matrix[adj_matrix >= t] = 1
    adj_matrix = adj_matrix * p / (adj_matrix.sum(0, keepdims=True) - 1 + 1e-6)

    # Ensure diagonal entries are 1 - p
    np.fill_diagonal(adj_matrix, 1.0 - p)

    # Save to pickle
    with open(adjacency_matrix_save_path, 'wb') as f:
        pickle.dump(adj_matrix, f)

def create_label_embeddings(rel2id, label_embeddings_save_path, model, tokenizer):
    model.eval()

    label_embeddings = []
    with torch.no_grad():  # Pas besoin de calculer les gradients
        for label in rel2id.keys():
            # Tokeniser le label
            inputs = tokenizer(label, return_tensors='pt')
            # Passer les inputs dans le modèle
            outputs = model(**inputs)
            # Prendre l'embedding de la dernière couche (tous les tokens) ou juste le [CLS] token
            # Utilisons l'embedding moyen ici pour chaque label
            label_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Moyenne sur les tokens
            label_embeddings.append(label_embedding)

    # Convertir la liste d'embeddings en un tensor
    label_embeddings = torch.stack(label_embeddings)

    with open(label_embeddings_save_path, 'wb') as f:
        pickle.dump(label_embeddings, f)
    
    return label_embeddings
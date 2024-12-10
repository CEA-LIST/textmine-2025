def get_rule_based_sent_labels(example, hts, sent_label):
    subject_entity = example['vertexSet'][hts[0]]  # Entité head
    object_entity = example['vertexSet'][hts[1]]   # Entité tail

    # Vérifier si les deux entités apparaissent dans la même phrase
    for k, sent_pos in enumerate(example['sent_pos']):
        for subject_mention in subject_entity:
            for object_mention in object_entity:
                if subject_mention['sent_id'] == k and object_mention['sent_id'] == k:
                    sent_label[k] = 1
                    break

    # 3. Méthode Bridge (voir https://aclanthology.org/2022.findings-acl.23.pdf section 3.3)
    # Si aucune co-occurrence directe n'est trouvée, on cherche une entité "bridge"
    if sent_label == [0] * len(example['sent_pos']):
        bridge_candidates = {}  # Dictionnaire pour stocker le nombre de co-occurrences pour chaque bridge

        # Parcourir toutes les entités candidates (différentes de head et tail)
        for idx, bridge_entity_candidate in enumerate(example['vertexSet']):
            if idx == hts[0] or idx == hts[1]:
                continue  # Ne pas prendre les entités head et tail comme bridge

            cooccurrence_count = 0  # Compte le nombre de phrases partagées avec head ou tail
            cooccurring_sentences = set()

            # Compter les phrases où l'entité bridge co-apparaît avec head ou tail
            for bridge_mention in bridge_entity_candidate:
                for subject_mention in subject_entity:
                    if bridge_mention['sent_id'] == subject_mention['sent_id']:
                        cooccurrence_count += 1
                        cooccurring_sentences.add(bridge_mention['sent_id'])
                        break
                for object_mention in object_entity:
                    if bridge_mention['sent_id'] == object_mention['sent_id']:
                        cooccurrence_count += 1
                        cooccurring_sentences.add(bridge_mention['sent_id'])
                        break

            # Enregistrer l'entité bridge et son nombre de co-occurrences
            if cooccurrence_count > 0:
                bridge_candidates[idx] = (cooccurrence_count, cooccurring_sentences)

        # Sélectionner les entités bridge avec le plus grand nombre de co-occurrences
        if bridge_candidates:
            # Trouver le nombre maximum de co-occurrences
            max_cooccurrence = max(bridge_candidates.values(), key=lambda x: x[0])[0]

            # Sélectionner toutes les entités bridge ayant le nombre maximum de co-occurrences
            best_bridge_sentences = set()
            for candidate, (count, sentences) in bridge_candidates.items():
                if count == max_cooccurrence:
                    best_bridge_sentences.update(sentences)

            # Marquer les phrases contenant les entités bridge sélectionnées comme preuves
            for sent_id in best_bridge_sentences:
                sent_label[sent_id] = 1
        
    # 4. Fallback : Si aucune coref ni bridge n'est trouvée, on prend toutes les phrases contenant une mention d'une des entités
    if sent_label == [0] * len(example['sent_pos']):
        for subject_mention in subject_entity:
            sent_label[subject_mention['sent_id']] = 1
        for object_mention in object_entity:
            sent_label[object_mention['sent_id']] = 1

    return sent_label


def get_sentence_labels(example, evidence_construction):
    sent_labels = []

    if evidence_construction == 'predictions_and_rule_based':
        for i, silver_evidence in enumerate(example['silver_evidence']):
            hts = example['hts'][i]
            sent_label = [0] * len(example['sent_pos'])
            
            # Utilisation des "silver evidences" si non vides
            if silver_evidence != []:
                for j in silver_evidence:
                    sent_label[j] = 1
            
            else:
                sent_label = get_rule_based_sent_labels(example, hts, sent_label)

            sent_labels.append(sent_label)

    elif evidence_construction == "rule_based":
        for i, hts in enumerate(example['hts']):
            sent_label = [0] * len(example['sent_pos'])
            sent_label = get_rule_based_sent_labels(example, hts, sent_label)

            sent_labels.append(sent_label)

    example['sent_labels'] = sent_labels
    return example

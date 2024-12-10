import pandas as pd
import spacy
import json
import re
import argparse

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *

def create_rel2id(df):
    """
    Crée un dictionnaire rel2id à partir des types de relations dans le DataFrame,
    en ajoutant 'UNRELATED' avec l'ID 0 et en attribuant des IDs aux autres relations à partir de 1.
    """
    # Récupérer tous les types de relations
    relation_types = set()
    for relations in df['relations']:
        for _, rel_type, _ in relations:
            relation_types.add(rel_type)
    
    # Créer un dictionnaire avec 'UNRELATED' = 0, et indexer les autres relations à partir de 1
    rel2id = {'UNRELATED': 0}
    rel2id.update({rel_type: idx for idx, rel_type in enumerate(sorted(relation_types), start=1)})
    
    return rel2id


def process_text(text, nlp):
    """
    Divise le texte en phrases et retourne une liste de tokens par phrase,
    ainsi qu'une liste des phrases fusionnées sous forme de texte.
    """
    doc = nlp(text)
    sentences = []
    merged_sentences_text = []
    current_sentence = []
    current_sentence_text = ""

    # Définir les ponctuations qui peuvent terminer une phrase
    end_of_sentence_punctuations = {".", "!", "?", "..."}

    for k, sent in enumerate(doc.sents):
        tokens = [token.text for token in sent]
        sentence_text = sent.text.strip()  # Texte brut de la phrase

        # Fusionner si la phrase commence par une virgule ou autre ponctuation faible
        if current_sentence and tokens[0] in {",", ";", ":", ".", "!", "?"}:
            current_sentence += tokens
            current_sentence_text += " " + sentence_text  # Ajout avec espace
            continue

        if current_sentence and current_sentence[-1] == "-":
            # Fusionner avec la phrase précédente et suivante (cas du tiret à la fin)
            if sentences and sentences[-1][-1] not in end_of_sentence_punctuations:
                sentences[-1] += ['-'] + tokens
                merged_sentences_text[-1] += current_sentence_text + sentence_text
            else:
                sentences.append(current_sentence + tokens)
                merged_sentences_text.append(current_sentence_text + sentence_text)
            current_sentence = []
            current_sentence_text = ''
            continue

        if current_sentence and current_sentence[0] == "-":
            # Fusionner avec la phrase précédente et suivante (cas du tiret au début)
            if sentences:
                sentences[-1] += current_sentence
                merged_sentences_text[-1] += current_sentence_text
            else:
                sentences.append(current_sentence)
                merged_sentences_text.append(current_sentence_text)
            current_sentence = tokens
            current_sentence_text = sentence_text
            continue

        # Vérifier si la phrase se termine par un caractère problématique
        if current_sentence and current_sentence[-1] in {"«", "(", '"', "'", ":"}:
            if current_sentence[-1] == "«":
                current_sentence += [" "] + tokens
                current_sentence_text += " " + sentence_text
            elif current_sentence[-1] == ':':
                current_sentence += tokens
                current_sentence_text += " " + sentence_text
            else:
                current_sentence += tokens
                current_sentence_text += sentence_text  
        else:
            # Vérifier si la phrase précédente ne se termine pas par une ponctuation forte
            if current_sentence and ((current_sentence[-2:] == ['Chimix', '.']) or (current_sentence[-1] not in end_of_sentence_punctuations and tokens[0] != '-')): 
                # Fusionner avec l'espace approprié
                if sentence_text in ['"', '.', '...'] or current_sentence[-2:] == ['Chimix', '.']:
                    current_sentence += tokens
                    current_sentence_text += sentence_text  
                elif sentence_text[0] == '-':
                    current_sentence[-1] += tokens[0] 
                    current_sentence += tokens[1:]
                    current_sentence_text += sentence_text  
                else:
                    current_sentence += tokens
                    current_sentence_text += " " + sentence_text
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    merged_sentences_text.append(current_sentence_text.strip())
                current_sentence = tokens
                current_sentence_text = sentence_text

    # Ajouter la dernière phrase après la boucle si elle est non vide
    if current_sentence:
        if current_sentence[0] == "-":
            sentences[-1] += current_sentence
            merged_sentences_text[-1] += current_sentence_text

        else:
            sentences.append(current_sentence)
            merged_sentences_text.append(current_sentence_text.strip())

    # Le reste du code reste inchangé
    # Vérification supplémentaire : ne pas diviser une phrase en cas de fin de sous-phrase
    final_merged_sentences = []
    final_merged_sentences_text = []
    
    for i, sentence in enumerate(sentences):
        if i > 0 and sentences[i - 1][-1].isalnum() and sentence[0][0].isupper():
            # Fusionner avec la précédente si nécessaire
            final_merged_sentences[-1] += sentence
            
            # Corriger les espaces uniquement entre les mots et les virgules
            merged_text = merged_sentences_text[i]
            merged_text = merged_text.replace(" ,", ",").replace(" .", ".")
            final_merged_sentences_text[-1] += " " + merged_text.lstrip()
        else:
            final_merged_sentences.append(sentence)
            final_merged_sentences_text.append(merged_sentences_text[i])

    # Correction finale pour les guillemets : enlever l'espace entre le mot et les guillemets fermants
    final_merged_sentences_text = [text.replace(" ,", ",").replace(" .", ".").replace(" )", ")")  for text in final_merged_sentences_text]
    final_merged_sentences_text = [text.replace('“ Raffinerie Rodriguez ”', '“Raffinerie Rodriguez”').replace('"Éradication épidémie africaine"ont', '"Éradication épidémie africaine" ont').replace('"France équipement chimique"s', '"France équipement chimique" s').replace('(longitude 11° 29′31,56″ Est)', '(longitude 11°29′31,56″ Est)').replace('"Paix Africaine pour Tous "', '"Paix Africaine pour Tous"').replace("‘ ’ Même vision’’", "‘’Même vision’’").replace('“ Ammoniac plus”', '“Ammoniac plus”')  for text in final_merged_sentences_text]
    final_merged_sentences_text = [text.replace('"Non au passe !"Le', '"Non au passe !" Le').replace('"Santé pour tous ",', '"Santé pour tous",').replace('"XCAO"à', '"XCAO" à').replace('“ FEAR”', '“FEAR”')  for text in final_merged_sentences_text]
    final_merged_sentences_text = [text.replace('"Hiver Arabe Universel "', '"Hiver Arabe Universel"').replace('"Santé pour tous ",', '"Santé pour tous",').replace('"XCAO"à', '"XCAO" à')  for text in final_merged_sentences_text]

    return final_merged_sentences, final_merged_sentences_text, doc


def find_sentence_id(start, sentences_text):
    """
    Trouve l'index de la phrase (sent_id) qui contient la mention, en utilisant les indices des caractères
    basés sur les phrases non tokenisées.
    """
    current_char_idx = 0  # Pour suivre l'indice du caractère dans le texte global

    for sent_id, sentence in enumerate(sentences_text):
        sentence_length = len(sentence)

        # Calculer les positions de début et de fin de la phrase dans le texte
        sentence_start_char = current_char_idx
        sentence_end_char = current_char_idx + sentence_length

        # if start == 147:
        #     print('tokens positions')
        #     print(sentence_start_char, sentence_end_char)

        # Vérifier si la mention commence dans cette phrase
        if sentence_start_char <= start < sentence_end_char:
            return sent_id
        
        # Mettre à jour l'indice de caractère pour la prochaine phrase
        current_char_idx += sentence_length + 1  # +1 pour l'espace ou la ponctuation entre les phrases

    return -1  # Si on ne trouve pas de correspondance


def get_token_positions(tokens):
    """ Retourne les positions de départ et de fin des tokens dans le texte. 
        Accepte directement une liste de tokens au lieu de text brut. 
    """
    position_mapping = []
    current_start = 0

    for token in tokens:
        token_start = current_start
        if len(token) == 1:
            token_end = token_start 
        else:
            token_end = token_start + len(token)
        
        position_mapping.append((token_start, token_end))
        current_start = token_end + 1  # Avancer après le token

    return tokens, position_mapping

def char_to_token_positions(start_char, end_char, token_positions, mention_value):
    """ Convertit les positions de caractères en positions de tokens dans une phrase donnée """
    start_token_pos = None
    end_token_pos = None
    mentions_split = mention_value.split()
    for i, (token_start, token_end) in enumerate(token_positions):
        if token_start == start_char and len(mentions_split[0]) == token_end - token_start:
            start_token_pos = i
        if end_char == token_end  and len(mentions_split[-1]) == token_end - token_start:
            end_token_pos = i + 1
        if start_token_pos is not None and end_token_pos is not None:
            return start_token_pos, end_token_pos
        
    # Créer la liste des offsets à vérifier dans l'ordre souhaité
    offsets = []
    for j in range(1, 6):
        offsets.append(j)    
        offsets.append(-j)   
    
    # bidouillage si on trouve pas avant
    if start_token_pos is None:
        for i, (token_start, token_end) in enumerate(token_positions):
            check_length = (len(mentions_split[0]) > 1 and len(mentions_split[0]) == token_end - token_start) or (len(mentions_split[0]) == 1 and len(mentions_split[0]) - 1 == token_end - token_start)

            for offset in offsets:  # Utilisez les offsets
                if token_start == start_char + offset and check_length:
                    start_token_pos = i
                    break  # Sortir de la boucle pour les offsets

            if start_token_pos is not None:
                break  # Sortir de la boucle principale si on a trouvé une position

    if end_token_pos is None:
        for i, (token_start, token_end) in enumerate(token_positions):
            check_length = (len(mentions_split[-1]) > 1 and len(mentions_split[-1]) == token_end - token_start) or (len(mentions_split[-1]) == 1 and len(mentions_split[-1]) - 1 == token_end - token_start)

            for offset in offsets:  # Utilisez les offsets
                if token_end == end_char + offset and check_length:
                    end_token_pos = i + 1  # Notez que vous pouvez ajuster ici selon vos besoins
                    break  # Sortir de la boucle pour les offsets

            if end_token_pos is not None:
                break  # Sortir de la boucle principale si on a trouvé une position

    #########################
    # si on n'a toujours pas trouvé, plus de check de longueur
    for i, (token_start, token_end) in enumerate(token_positions):
        if token_start == start_char:
            start_token_pos = i
        if end_char == token_end:
            end_token_pos = i + 1
        if start_token_pos is not None and end_token_pos is not None:
            # print(start_token_pos, end_token_pos)
            return start_token_pos, end_token_pos
        
    # Créer la liste des offsets à vérifier dans l'ordre souhaité
    offsets = []
    for j in range(1, 6):
        offsets.append(j)    
        offsets.append(-j)   
    
    # bidouillage si on trouve pas avant
    if start_token_pos is None:
        for i, (token_start, token_end) in enumerate(token_positions):
            for offset in offsets:  # Utilisez les offsets
                if token_start == start_char + offset:
                    start_token_pos = i
                    break  # Sortir de la boucle pour les offsets

            if start_token_pos is not None:
                break  # Sortir de la boucle principale si on a trouvé une position

    if end_token_pos is None:
        for i, (token_start, token_end) in enumerate(token_positions):
            for offset in offsets:  # Utilisez les offsets
                if token_end == end_char + offset:
                    end_token_pos = i + 1  # Notez que vous pouvez ajuster ici selon vos besoins
                    break  # Sortir de la boucle pour les offsets

            if end_token_pos is not None:
                break  # Sortir de la boucle principale si on a trouvé une position

    # manual fixes for entities which are part of a composed word (start_token_pos cant be found as a composed word is one token)
    if start_token_pos is None and mention_value=="enfants":
        start_token_pos = 1
    elif start_token_pos is None and mention_value=="Corse" and end_token_pos == 16:
        start_token_pos = 15
    elif start_token_pos is None and mention_value=="Corse" and end_token_pos == 17:
        start_token_pos = 15
    return start_token_pos, end_token_pos


def count_apostrophes(input_string):
    # Compte les apostrophes simples et typographiques
    standard_apostrophe_count = input_string.count("'")
    typographic_apostrophe_count = input_string.count("’")  # Apostrophe typographique
    
    total_count = standard_apostrophe_count + typographic_apostrophe_count
    return total_count


def create_vertex_set(entities, text, sentences_tokenized, sentences_text, doc):
    """
    Crée la liste vertex_set contenant les entités, leurs positions de tokens relatives à chaque phrase (sent_id).
    """

    vertex_set = []
    for ent in entities:
        entity_list = []
        for mention in ent['mentions']:
            start_char = mention['start']
            end_char = mention['end']
            mention_value = mention['value']
            entity_type = ent['type']

            # Trouver l'ID de la phrase correspondant à cette mention
            sent_id = find_sentence_id(start_char, sentences_text)

            if sent_id != -1:
                # Tokenisation de la phrase actuelle (basée sur notre découpage)
                sentence_tokens = sentences_tokenized[sent_id]

                # Trouver la phrase sous forme de texte pour ajuster les indices
                sentence_text = sentences_text[sent_id]

                
                # sentence_start_char = text.find(sentence_text)
                matches = list(re.finditer(sentence_text, text))
                matches = list(re.finditer(re.escape(sentence_text), text))  # re.escape() pour traiter les caractères spéciaux
                if len(matches) > 1 and 'Ly Liu' in sentence_text and start_char > 127:
                    sentence_start_char = matches[1].start()
                else:
                    sentence_start_char = matches[0].start()


                # Ajuster les indices de la mention pour la phrase actuelle
                start_char_in_sent = start_char - sentence_start_char
                end_char_in_sent = end_char - sentence_start_char
                # Obtenir les positions des tokens dans la phrase actuelle
                _, sentence_token_positions = get_token_positions(sentence_tokens)

                # Trouver les positions des tokens dans cette phrase
                # apostrophe_count = count_apostrophes(mention_value)
                start_token_pos, end_token_pos = char_to_token_positions(start_char_in_sent, end_char_in_sent, sentence_token_positions, mention_value)
                if start_token_pos is None or end_token_pos is None:
                    print('TOKEN POS NOT FOUND')
                

                if start_token_pos is not None and end_token_pos is not None:
                    entity_list.append({
                        'name': mention_value,
                        'pos': [start_token_pos, end_token_pos],
                        'sent_id': sent_id,
                        'type': entity_type
                    })
        
        if entity_list:
            vertex_set.append(entity_list)

    return vertex_set

def create_labels(relations):
    """
    Crée les 'labels' en fonction des relations entre les entités.
    """
    labels = []
    for relation in relations:
        head_id, rel_type, tail_id = relation
        labels.append({
            'h': head_id,
            't': tail_id,
            'r': rel_type,
            'evidence': []
        })
    return labels


def convert_to_docred_format(example, nlp):
    """
    Convertit un exemple en un format adapté à DocRED avec les colonnes 'vertexSet', 'labels', 'title', 'sents'.
    """
    text = example['text']
    text = text.replace('\xa0', ' ')
    example['text'] = text

    entities = example['entities']

    title = example.name  # Utiliser l'index comme titre

    # 1. Diviser le texte en phrases et tokeniser
    sentences_tokenized, sentences_text, doc = process_text(text, nlp)

    # no other way to fix it 
    if example.name == "2_1418":
        sentences_tokenized = [['Depuis', 'plusieurs', 'jours', ',', 'le', 'scandale', 'lié', 'à', "l'", 'augmentation', 'des', 'prix', 'des', 'produits', 'médicaux', 'en', 'Birmanie', 'provoquée', 'par', "l'", 'organisation', 'BIRMAR', "n'", 'a', 'pas', 'faibli', '.'], ['Cette', 'organisation', ',', 'qui', 'opère', 'dans', 'le', 'domaine', 'de', 'la', 'santé', 'publique', 'en', 'Birmanie', ',', 'a', 'été', 'dissoute', 'le', '23', 'avril', 'dernier', ',', 'mais', 'les', 'conséquences', 'de', 'ses', 'actions', 'restent', 'encore', 'visibles', '.'], ['Pendant', 'que', "l'", 'épidémie', 'se', 'poursuit', ',', 'le', 'scandale', 'fait', 'rage', 'et', 'ne', 'semble', 'pas', "s'", 'arrêter', '.'], ['Zaïra', 'Jackson', ',', 'employée', 'de', "l'", 'association', 'Urgence', '-', 'Santé', ',', 'voit', 'quotidiennement', 'les', 'conséquences', 'de', 'cet', 'incident', '.'], ['Elle', "s'", 'efforce', 'avec', 'ses', 'collègues', 'de', 'faire', 'face', 'à', 'la', 'panne', 'causée', 'par', 'BIRMAR', 'dans', 'les', 'laboratoires', 'birmans', ',', 'ainsi', "qu'", 'à', "l'", 'épidémie', 'qui', 'en', 'découle', '.'], ["L'", 'organisation', 'BIRMAR', 'contrôle', 'non', 'seulement', 'les', 'laboratoires', 'où', 'sont', 'fabriqués', 'les', 'isolateurs', ',', 'filtres', 'à', 'particules', 'et', 'autres', 'équipements', 'essentiels', ',', 'mais', 'elle', 'a', 'également', 'initié', "l'", 'épidémie', ',', 'la', 'panne', 'et', 'le', 'scandale', 'qui', 'secouent', 'le', 'pays', '.']]
        sentences_text = ["Depuis plusieurs jours, le scandale lié à l'augmentation des prix des produits médicaux en Birmanie provoquée par l'organisation BIRMAR n'a pas faibli.", 'Cette organisation, qui opère dans le domaine de la santé publique en Birmanie, a été dissoute le 23 avril dernier, mais les conséquences de ses actions restent encore visibles.', "Pendant que l'épidémie se poursuit, le scandale fait rage et ne semble pas s'arrêter.", "Zaïra Jackson, employée de l'association Urgence-Santé, voit quotidiennement les conséquences de cet incident.", "Elle s'efforce avec ses collègues de faire face à la panne causée par BIRMAR dans les laboratoires birmans, ainsi qu'à l'épidémie qui en découle.", "L'organisation BIRMAR contrôle non seulement les laboratoires où sont fabriqués les isolateurs, filtres à particules et autres équipements essentiels, mais elle a également initié l'épidémie, la panne et le scandale qui secouent le pays."]

    # 2. Créer le vertexSet
    vertex_set = create_vertex_set(entities, text, sentences_tokenized, sentences_text, doc)

    # 3. Créer les labels
    if "relations" in example.index:
        relations = example['relations']
        labels = create_labels(relations)
        example['labels_for_atlop'] = labels    

    # 4. Ajouter les données formatées
    example['title'] = title
    example['sents'] = sentences_tokenized
    example['vertexSet'] = vertex_set

    return example



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", required=True, type=str, help="Path to the project folder")
    parser.add_argument("--data_file", required=True, type=str, help="File name")
    parser.add_argument("--save_path", required=True, type=str, help="Path where the formatted data will be saved")
    args = parser.parse_args()

    dataset_path = f"{args.project_path}/data/{args.data_file}"
    df = load_data(dataset_path)

    # Charger le modèle de spaCy en français
    nlp = spacy.load('fr_core_news_lg')

    df = df.apply(lambda row: convert_to_docred_format(row, nlp), axis=1)

    os.makedirs(args.save_path, exist_ok=True)

    if args.data_file == "train.csv":
        rel2id = create_rel2id(df)
        with open(f'{args.save_path}/rel2id.json', 'w') as f:
            json.dump(rel2id, f, indent=4)

        # Sauvegarder les datasets au format JSON
        data_file = args.data_file.replace('.csv', '.json')
        df.to_json(f'{args.save_path}/{data_file}', orient='records', force_ascii=False, indent=4)

    elif args.data_file == "test_01-07-2024.csv":
        data_file = args.data_file.replace('.csv', '.json')
        df.to_json(f'{args.save_path}/{data_file}', orient='records', force_ascii=False, indent=4)

    elif args.data_file == "synthetic.json":
        data_file = args.data_file.replace('.csv', '.json')
        df.to_json(f'{args.save_path}/{data_file}', orient='records', force_ascii=False, indent=4)
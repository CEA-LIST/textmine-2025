import pandas as pd
import json
from collections import defaultdict
import os
from collections import defaultdict
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from utils.ontology import *

SUBJECT_RELATION_OBJECT_TABLE = [
    ("ACTOR, EVENT, PLACE", "IS_LOCATED_IN", "PLACE"),
    ("ACTOR, PLACE", "IS_OF_NATIONALITY", "NATIONALITY"),
    ("ACTOR", "CREATED", "ORGANIZATION"),
    ("ACTOR", "HAS_CONTROL_OVER", "ACTOR, MATERIAL, PLACE"),
    ("ACTOR", "INITIATED", "EVENT"),
    ("ACTOR", "IS_AT_ODDS_WITH", "ACTOR"),
    ("ACTOR", "IS_COOPERATING_WITH", "ACTOR"),
    ("ACTOR", "IS_IN_CONTACT_WITH", "ACTOR"),
    ("ACTOR", "IS_PART_OF", "ORGANIZATION"),
    ("EVENT", "DEATH_NUMBER", "QUANTITY"),
    ("EVENT", "END_DATE", "TIME"),
    ("EVENT", "HAS_CONSEQUENCE", "EVENT"),
    ("EVENT", "INJURED_NUMBER", "QUANTITY"),
    ("EVENT", "START_DATE", "TIME"),
    ("EVENT", "STARTED_IN", "PLACE"),
    ("MATERIAL", "HAS_COLOR", "COLOR"),
    ("MATERIAL", "HAS_FOR_HEIGHT", "HEIGHT"),
    ("MATERIAL", "HAS_FOR_LENGTH", "LENGTH"),
    ("MATERIAL", "HAS_FOR_WIDTH", "WIDTH"),
    ("MATERIAL", "HAS_QUANTITY", "QUANTITY"),
    ("MATERIAL", "IS_REGISTERED_AS", "MATERIAL_REFERENCE"),
    ("MATERIAL", "WEIGHS", "WEIGHT"),
    ("ORGANIZATION", "CREATED_IN", "TIME"),
    ("ORGANIZATION", "DISSOLVED_IN", "TIME"),
    ("ORGANIZATION", "IS_OF_SIZE", "QUANTITY"),
    ("ORGANIZATION", "OPERATES_IN", "PLACE"),
    ("PERSON", "DIED_IN", "EVENT"),
    ("PERSON", "HAS_CATEGORY", "CATEGORY"),
    ("PERSON", "HAS_FAMILY_RELATIONSHIP", "PERSON"),
    ("PERSON", "HAS_FIRST_NAME", "FIRSTNAME"),
    ("PERSON", "HAS_GENDER_FEMALE", "N/A"),
    ("PERSON", "HAS_GENDER_MALE", "N/A"),
    ("PERSON", "HAS_LAST_NAME", "LASTNAME"),
    ("PERSON", "IS_BORN_IN", "PLACE"),
    ("PERSON", "IS_BORN_ON", "TIME"),
    ("PERSON", "IS_DEAD_ON", "TIME"),
    ("PERSON", "RESIDES_IN", "PLACE"),
    ("PLACE", "HAS_LATITUDE", "LATITUDE"),
    ("PLACE", "HAS_LONGITUDE", "LONGITUDE"),
]


SUBTYPE_TO_GENERAL = {
    # Actor types
    "GOVERNMENT_ORGANIZATION": "ORGANIZATION",
    "MILITARY_ORGANISATION": "ORGANIZATION",
    "NON_MILITARY_GOVERNMENT_ORGANISATION": "ORGANIZATION",
    "GROUP_OF_INDIVIDUALS": "ORGANIZATION",
    "INTERGOVERNMENTAL_ORGANISATION": "ORGANIZATION",
    "NON_GOVERNMENTAL_ORGANISATION": "ORGANIZATION",
    "CIVILIAN": "PERSON",
    "TERRORIST_OR_CRIMINAL": "PERSON",
    "MILITARY": "PERSON",
    
    # Organization maps to Actor
    "ORGANIZATION": "ACTOR",
    "PERSON": "ACTOR",
    
    # Event types
    "ACCIDENT": "EVENT",
    "CBRN_EVENT": "EVENT",
    "AGITATING_TROUBLE_MAKING": "EVENT",
    "CIVIL_WAR_OUTBREAK": "EVENT",
    "COUP_D_ETAT": "EVENT",
    "DEMONSTRATION": "EVENT",
    "ELECTION": "EVENT",
    "GATHERING": "EVENT",
    "ILLEGAL_CIVIL_DEMONSTRATION": "EVENT",
    "NATURAL_CAUSES_DEATH": "EVENT",
    "RIOT": "EVENT",
    "STRIKE": "EVENT",
    "SUICIDE": "EVENT",
    "BOMBING": "EVENT",
    "CRIMINAL_ARREST": "EVENT",
    "DRUG_OPERATION": "EVENT",
    "HOOLIGANISM_TROUBLEMAKING": "EVENT",
    "POLITICAL_VIOLENCE": "EVENT",
    "THEFT": "EVENT",
    "TRAFFICKING": "EVENT",
    "ECONOMICAL_CRISIS": "EVENT",
    "EPIDEMIC": "EVENT",
    "FIRE": "EVENT",
    "NATURAL_EVENT": "EVENT",
    "POLLUTION": "EVENT",
    
    # Quantity attributes
    "QUANTITY_EXACT": "QUANTITY",
    "QUANTITY_FUZZY": "QUANTITY",
    "QUANTITY_MAX": "QUANTITY",
    "QUANTITY_MIN": "QUANTITY",

    # Time attributes
    "TIME_EXACT": "TIME",
    "TIME_FUZZY": "TIME",
    "TIME_MAX": "TIME",
    "TIME_MIN": "TIME"
}

SUBJECT_TYPES = ['ACTOR', 'PERSON', 'ORGANIZATION', 'EVENT', 'PLACE', 'MATERIAL']
OBJECT_TYPES = ['PLACE', 'NATIONALITY', 'ORGANIZATION', 'ACTOR', 'MATERIAL', 'EVENT', 'QUANTITY', 'TIME', 
                'COLOR', 'HEIGHT', 'LENGTH', 'WIDTH', 'MATERIAL_REFERENCE', 'WEIGHT', 'CATEGORY', 'PERSON', 
                'FIRSTNAME', 'N/A', 'LASTNAME', 'LATITUDE', 'LONGITUDE']


def lower_case_str(item):
    # Si l'item est une chaîne, on la met en minuscule et on remplace les underscores
    if isinstance(item, str):
        return item.lower().replace('_', ' ')
    return item

def transform_relations(relations):
    return [[lower_case_str(i) for i in relation] for relation in relations]

def transform_entities(entities):
    for entity in entities:
        if 'type' in entity:
            entity['type'] = lower_case_str(entity['type'])
    return entities

def lower_case_labels(row):
    row['relations'] = transform_relations(row['relations'])
    row['entities'] = transform_entities(row['entities'])
    return row

def get_subject_relation_object_dicts(lower_case_labels=False):

    # Création du premier objet
    subject_to_relations_dict = defaultdict(list)

    # Création du deuxième objet
    subject_object_to_relations_dict = defaultdict(list)

    for subjects, relation, objects in SUBJECT_RELATION_OBJECT_TABLE:
        # Diviser les types de sujets et d'objets multiples
        SUBJECT_TYPES = [s.strip() for s in subjects.split(",")]
        OBJECT_TYPES = [o.strip() for o in objects.split(",")]

        # Transformer les relations si nécessaire
        if lower_case_labels:
            relation = lower_case_str(relation)

        # Remplir le premier dictionnaire
        for subject in SUBJECT_TYPES:
            if lower_case_labels:
                subject = lower_case_str(subject)
            if relation not in subject_to_relations_dict[subject]:
                subject_to_relations_dict[subject].append(relation)

        # Remplir le deuxième dictionnaire
        for subject in SUBJECT_TYPES:
            if lower_case_labels:
                subject = lower_case_str(subject)
            for obj in OBJECT_TYPES:
                if lower_case_labels:
                    obj = lower_case_str(obj)
                if relation not in subject_object_to_relations_dict[(subject, obj)]:
                    subject_object_to_relations_dict[(subject, obj)].append(relation)

    # Conversion de defaultdict en dict pour un usage plus général
    subject_to_relations_dict = dict(subject_to_relations_dict)
    subject_object_to_relations_dict = dict(subject_object_to_relations_dict)
    return subject_to_relations_dict, subject_object_to_relations_dict


def lowercase_lists(subtype_to_general, subject_types, object_types):
    # Transformer le dictionnaire SUBTYPE_TO_GENERAL
    transformed_subtype_to_general = {lower_case_str(k): lower_case_str(v) for k, v in subtype_to_general.items()}

    # Transformer les listes SUBJECT_TYPES et OBJECT_TYPES
    transformed_subject_types = [lower_case_str(s) for s in subject_types]
    transformed_object_types = [lower_case_str(o) for o in object_types]

    return transformed_subtype_to_general, transformed_subject_types, transformed_object_types


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.set_index("id")
    df.entities = df.entities.apply(json.loads)
    if 'relations' in df.columns:
        df.relations = df.relations.apply(json.loads)
    return df

# Fonction pour obtenir les occurrences et proportions des classes dans chaque split
def get_split_proportion_per_class(train_df, dev_df, test_df, classes):
    # Calculer les occurrences totales par classe dans l'ensemble des données
    total_counts = {cls: train_df[cls].sum() + dev_df[cls].sum() + test_df[cls].sum() for cls in classes}

    # Calculer les occurrences et proportions pour chaque split
    splits = {'Train': train_df, 'Dev': dev_df, 'Test': test_df}
    results = {}

    for split_name, df_split in splits.items():
        split_counts = {cls: df_split[cls].sum() for cls in classes}  # Occurrences par classe dans le split
        split_proportions = {cls: (split_counts[cls] / total_counts[cls] * 100) if total_counts[cls] > 0 else 0
                            for cls in classes}  # Proportions par classe

        results[split_name] = split_proportions

    # Affichage des résultats
    for cls in classes:
        print(f"\nClasse '{cls}':")
        for split_name in splits.keys():
            print(f" - {split_name}: {results[split_name][cls]:.2f}% des occurrences")

def remove_class_columns(df, all_classes):
    return df.drop(columns=all_classes)
    
# def split_data(df):
#     train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
#     dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
#     return train_df, dev_df, test_df

def split_data(df):
    all_classes = sorted(list(set(relation[1] for relations in df['relations'] for relation in relations)))

    # Créer une colonne pour chaque classe avec le nombre d'occurrences de la classe pour chaque exemple
    for cls in all_classes:
        df[cls] = df['relations'].apply(lambda x: sum(1 for relation in x if relation[1] == cls))

    # Splitter les données en train, dev, test
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(msss.split(df, df[all_classes].values))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]
    msss_temp = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    dev_idx, test_idx = next(msss_temp.split(temp_df, temp_df[all_classes].values))
    dev_df, test_df = temp_df.iloc[dev_idx], temp_df.iloc[test_idx]

    # # Appeler la fonction pour obtenir les proportions
    # get_split_proportion_per_class(train_df, dev_df, test_df, all_classes)

    train_df= remove_class_columns(train_df, all_classes)
    dev_df = remove_class_columns(dev_df, all_classes)
    test_df = remove_class_columns(test_df, all_classes)

    return train_df, dev_df, test_df

def check_if_mentions_in_sentence(sentence, mention_1, mention_2):
    sent_start = sentence.start_char
    sent_end = sentence.end_char
    # Vérifier si mention_1 et mention_2 sont dans cette phrase
    if mention_1['start'] >= sent_start and mention_1['end'] <= sent_end and mention_2['start'] >= sent_start and mention_2['end'] <= sent_end:
        return True
    else:
        return False
    

    
# def get_potential_relations(entity_1, entity_2, relation):
#     original_subject_type = entity_1['type']
#     original_object_type = entity_2['type']
#     # types can be part of a more general type (those present in subject_object_to_relations_dict)
#     # relations are constructed with more and more general types, starting from the most specific types (those presents in example['entities']) 
#     subject_type = original_subject_type
#     potential_relation_types_list = []
#     while subject_type:
#         if subject_type in SUBJECT_TYPES:
#             object_type = original_object_type
#             while object_type:
#                 if object_type in OBJECT_TYPES:
#                     potential_relation_types = subject_object_to_relations_dict.get((subject_type, object_type))
#                     if potential_relation_types is not None:
#                         potential_relation_types_list.append(potential_relation_types)
#                 object_type = SUBTYPE_TO_GENERAL.get(object_type)
#         subject_type = SUBTYPE_TO_GENERAL.get(subject_type)
#     potential_relation_types_list = [item for sublist in potential_relation_types_list for item in sublist]
#     return potential_relation_types_list
    
def get_potential_relations(entity_1, entity_2):
    if isinstance(entity_1, list):
        subject_type = entity_1[0]['type']
        object_type = entity_2[0]['type']
    else:
        subject_type = entity_1['type']
        object_type = entity_2['type']
    parent_subject_type = TYPE_TO_PARENT.get(subject_type)
    parent_object_type = TYPE_TO_PARENT.get(object_type)
    potential_relations = []
    for relation in VALID_COMBINATIONS:
        if relation[0] == parent_subject_type and relation[2] == parent_object_type:
            potential_relations.append(relation[1])
            
    return potential_relations

# def check_if_relation_is_possible(entity_1, entity_2):
#     original_subject_type = entity_1['type']
#     original_object_type = entity_2['type']
#     # types can be part of a more general type (those present in subject_object_to_relations_dict)
#     # relations are constructed with more and more general types, starting from the most specific types (those presents in example['entities']) 
#     subject_type = original_subject_type
#     while subject_type:
#         if subject_type in SUBJECT_TYPES:
#             object_type = original_object_type
#             while object_type:
#                 if object_type in OBJECT_TYPES:
#                     potential_relation_types = subject_object_to_relations_dict.get((subject_type, object_type))
#                     if potential_relation_types is not None:
#                         return True
#                 object_type = SUBTYPE_TO_GENERAL.get(object_type)
#         subject_type = SUBTYPE_TO_GENERAL.get(subject_type)
#     return False

def check_if_relation_is_possible(entity_1, entity_2):
    subject_type = entity_1['type']
    object_type = entity_2['type']
    parent_subject_type = TYPE_TO_PARENT.get(subject_type)
    parent_object_type = TYPE_TO_PARENT.get(object_type)
    for relation in VALID_COMBINATIONS:
        if relation[0] == parent_subject_type and relation[2] == parent_object_type:
            return True            
    return False
 
def calculate_f1_scores_for_dataframe(df, column_preds):
    all_true_positives = defaultdict(int)
    all_false_positives = defaultdict(int)
    all_false_negatives = defaultdict(int)
    all_labels = set()
    total_support = defaultdict(int)

    for _, row in df.iterrows():
        gold_relations = row["relations"]
        predicted_relations = row[column_preds]
        # print('---------------------')
        # print(gold_relations)
        # print('#################"')
        # print(predicted_relations)

        # Convert relations to sets for easier comparison
        gold_set = set(tuple(rel) for rel in gold_relations)
        predicted_set = set(tuple(rel) for rel in predicted_relations)

        # Collect all labels from both gold and predicted sets
        for _, rel, _ in gold_set:
            all_labels.add(rel)
        for _, rel, _ in predicted_set:
            all_labels.add(rel)

        # Initialize counters for the current row
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        # Compute TP, FP for predicted relations
        for (s1, rel, e1) in predicted_set:
            if (s1, rel, e1) in gold_set:
                true_positives[rel] += 1
            else:
                false_positives[rel] += 1

        # Compute FN for missed gold relations
        for (s1, rel, e1) in gold_set:
            if (s1, rel, e1) not in predicted_set:
                false_negatives[rel] += 1

        # Accumulate results for each label
        for label in all_labels:
            all_true_positives[label] += true_positives[label]
            all_false_positives[label] += false_positives[label]
            all_false_negatives[label] += false_negatives[label]
            total_support[label] += (true_positives[label] + false_negatives[label])

    # Calculate scores for each class
    labels = sorted(all_labels)
    precision = []
    recall = []
    f1_score = []
    support = []

    for label in labels:
        tp = all_true_positives[label]
        fp = all_false_positives[label]
        fn = all_false_negatives[label]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
        support.append(total_support[label])

    # Calculate micro scores
    total_tp = sum(all_true_positives.values())
    total_fp = sum(all_false_positives.values())
    total_fn = sum(all_false_negatives.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Calculate macro scores (average of per-class scores)
    macro_precision = sum(precision) / len(precision) if len(precision) > 0 else 0
    macro_recall = sum(recall) / len(recall) if len(recall) > 0 else 0
    macro_f1 = sum(f1_score) / len(f1_score) if len(f1_score) > 0 else 0

    # Format the results
    evaluation_results = {
        'scores_by_class': {
            'target_names': labels,
            'precision': precision,
            'recall': recall,
            'f_score': f1_score,
            'true_sum': support
        },
        'micro_scores': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f_score': micro_f1
        },
        'macro_scores': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f_score': macro_f1
        }
    }
    
    return evaluation_results


def aggregate_kfold_results(evaluation_results_list):
    # Rassembler toutes les classes uniques
    all_classes = set()
    for results in evaluation_results_list:
        all_classes.update(results['scores_by_class']['target_names'])
    
    all_classes = sorted(all_classes)  # Tri des classes pour garantir un ordre cohérent

    # Initialiser des dictionnaires pour stocker les résultats
    precision_list = []
    recall_list = []
    f_score_list = []
    true_sum_list = []
    
    micro_precision_list = []
    micro_recall_list = []
    micro_f_score_list = []
    
    macro_precision_list = []
    macro_recall_list = []
    macro_f_score_list = []
    
    # Itérer sur chaque dictionnaire d'évaluation
    for results in evaluation_results_list:
        scores_by_class = results['scores_by_class']
        micro_scores = results['micro_scores']
        macro_scores = results['macro_scores']
        
        # Initialiser des tableaux avec NaN
        precision = np.full(len(all_classes), np.nan)
        recall = np.full(len(all_classes), np.nan)
        f_score = np.full(len(all_classes), np.nan)
        true_sum = np.full(len(all_classes), np.nan)
        
        # Remplir les tableaux avec les scores existants
        for i, class_name in enumerate(scores_by_class['target_names']):
            if class_name in all_classes:
                index = all_classes.index(class_name)
                precision[index] = scores_by_class['precision'][i]
                recall[index] = scores_by_class['recall'][i]
                f_score[index] = scores_by_class['f_score'][i]
                true_sum[index] = scores_by_class['true_sum'][i]

        precision_list.append(precision)
        recall_list.append(recall)
        f_score_list.append(f_score)
        true_sum_list.append(true_sum)
        
        micro_precision_list.append(micro_scores['precision'])
        micro_recall_list.append(micro_scores['recall'])
        micro_f_score_list.append(micro_scores['f_score'])
        
        macro_precision_list.append(macro_scores['precision'])
        macro_recall_list.append(macro_scores['recall'])
        macro_f_score_list.append(macro_scores['f_score'])
    
    # Calculer les moyennes en ignorant les NaN
    aggregated_results = {
        'scores_by_class': {
            'target_names': all_classes,
            'precision': np.nanmean(precision_list, axis=0).tolist(),
            'recall': np.nanmean(recall_list, axis=0).tolist(),
            'f_score': np.nanmean(f_score_list, axis=0).tolist(),
            'true_sum': np.nansum(true_sum_list, axis=0).tolist()  # Somme des supports
        },
        'micro_scores': {
            'precision': np.mean(micro_precision_list),
            'recall': np.mean(micro_recall_list),
            'f_score': np.mean(micro_f_score_list)
        },
        'macro_scores': {
            'precision': np.mean(macro_precision_list),
            'recall': np.mean(macro_recall_list),
            'f_score': np.mean(macro_f_score_list)
        }
    }
    
    return aggregated_results


def format_results(evaluation_results):
    scores_by_class = evaluation_results['scores_by_class']
    micro_scores = evaluation_results['micro_scores']
    macro_scores = evaluation_results['macro_scores']

    output_str = f"{'Label':<30}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}\n"
    output_str += "-" * 100 + "\n"

    for label, precision, recall, f_score, support in zip(scores_by_class['target_names'], scores_by_class['precision'], scores_by_class['recall'], scores_by_class['f_score'], scores_by_class['true_sum']):
        output_str += f"{label:<30}{precision:<15.4f}{recall:<15.4f}{f_score:<15.4f}{int(support):<10}\n"

    output_str += "-" * 100 + "\n"
    output_str += f"{'TOTAL (micro)':<30}{micro_scores['precision']:<15.4f}{micro_scores['recall']:<15.4f}{micro_scores['f_score']:<15.4f}{int(sum(scores_by_class['true_sum'])):<10}\n"
    output_str += f"{'TOTAL (macro)':<30}{macro_scores['precision']:<15.4f}{macro_scores['recall']:<15.4f}{macro_scores['f_score']:<15.4f}{int(sum(scores_by_class['true_sum'])):<10}\n"

    return output_str


def format_for_submission(df):
    df = df[['formatted_predictions']]
    df = df.rename(columns={'formatted_predictions': 'relations'})
    return df
    
def format_and_save_for_submission(df, pred_path):
    df = format_for_submission(df)
    df['relations'] = df['relations'].apply(lambda x: str(x).replace("'", '"'))
    df.to_csv(pred_path.replace('.json', '.csv'))


def save_arguments(args):
    # Enregistrer les arguments dans un fichier texte
    with open(os.path.dirname(args.pred_path) + '/arguments.txt', "w") as f:
        for arg in vars(args):
            # Ecriture de la clé et de la valeur au format "clé = valeur"
            f.write(f"{arg} = {getattr(args, arg)}\n")


def get_folds_for_cross_validation(df, args):
    # Compter les occurrences des relations dans le dataset complet
    relation_counts = defaultdict(int)
    for relations in df['relations']:
        for _, relation_type, _ in relations:
            relation_counts[relation_type] += 1

    # Initialiser les folds
    K = args.num_folds
    folds = [[] for _ in range(K)]
    fold_relation_counts = [defaultdict(int) for _ in range(K)]

    # Répartition manuelle des exemples dans les folds
    for idx, relations in enumerate(df['relations']):
        # Trouver le fold avec le moins d'occurrences des relations présentes dans cet exemple
        best_fold = None
        min_fold_score = float('inf')
        
        for fold_idx in range(K):
            fold_score = 0
            for _, relation_type, _ in relations:
                fold_score += fold_relation_counts[fold_idx][relation_type]
            if fold_score < min_fold_score:
                min_fold_score = fold_score
                best_fold = fold_idx

        # Ajouter l'exemple au meilleur fold trouvé
        folds[best_fold].append(idx)
        for _, relation_type, _ in relations:
            fold_relation_counts[best_fold][relation_type] += 1

    return folds
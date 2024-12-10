import pandas as pd
import json
from collections import defaultdict
import os
from collections import defaultdict
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from utils.ontology import *

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.set_index("id")
    df.entities = df.entities.apply(json.loads)
    if 'relations' in df.columns:
        df.relations = df.relations.apply(json.loads)
    return df

def remove_class_columns(df, all_classes):
    return df.drop(columns=all_classes)
    
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

    train_df= remove_class_columns(train_df, all_classes)
    dev_df = remove_class_columns(dev_df, all_classes)
    test_df = remove_class_columns(test_df, all_classes)

    return train_df, dev_df, test_df

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
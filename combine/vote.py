import pandas as pd
import pandas as pd
from ast import literal_eval
import argparse
from itertools import combinations
from collections import Counter, defaultdict


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_atlop_lace import load_data
from utils.utils import calculate_f1_scores_for_dataframe
from utils.utils import format_results


def get_combinations_probabilities(args):

    df = load_data(args.train_data_path)

    all_relations = []

    for rel_list in df['relations']:
        entity_pairs = {}
        for subject, rel_type, obj in rel_list:
            pair = (subject, obj)
            if pair not in entity_pairs:
                entity_pairs[pair] = []
            entity_pairs[pair].append(rel_type)
        all_relations.append(entity_pairs)

    # Unir tous les couples d'entités dans une seule structure
    combined_entity_pairs = {}
    for pairs in all_relations:
        for pair, relations in pairs.items():
            if pair not in combined_entity_pairs:
                combined_entity_pairs[pair] = []
            combined_entity_pairs[pair].extend(relations)

    # Identifier toutes les relations uniques
    all_unique_relations = set()
    for relations in combined_entity_pairs.values():
        all_unique_relations.update(relations)
    all_unique_relations = sorted(all_unique_relations)  # Pour un ordre constant

    comb_counts = {2: Counter(), 3: Counter(), 4: Counter()}

    # Calculer les combinaisons de relations présentes dans les données
    for relations in combined_entity_pairs.values():
        for k in range(2, 5):  # Pour les combinaisons de 2, 3 et 4 relations
            for combo in combinations(sorted(set(relations)), k):
                comb_counts[k][combo] += 1

    # Générer toutes les combinaisons possibles de relations et leur assigner 0 si elles n'apparaissent pas
    for k in range(2, 5):
        all_combos = list(combinations(all_unique_relations, k))
        for combo in all_combos:
            if combo not in comb_counts[k]:
                comb_counts[k][combo] = 0

    comb_probabilities = {}

    for k, counts in comb_counts.items():
        total_combos = sum(counts.values())
        if total_combos == 0:
            # Si aucune combinaison de cette taille n'existe, assigner une probabilité de 0 à toutes les combinaisons
            comb_probabilities[k] = {combo: 0 for combo in counts.keys()}
        else:
            # Sinon, calculer la probabilité de chaque combinaison
            comb_probabilities[k] = {combo: count / total_combos for combo, count in counts.items()}
        
    return comb_probabilities


def get_absolute_majority_predictions(dfs, column):
    votes = {}
    for df in dfs:
        for _, row in df.iterrows():
            doc_id = row['id']
            relations = row[column]
            
            if doc_id not in votes:
                votes[doc_id] = []
            
            # Ajouter toutes les relations pour ce modèle au vote pour cet id
            votes[doc_id].extend(relations)

    final_predictions = {}

    for doc_id, relations in votes.items():
        # Compter chaque relation
        relation_counts = Counter(map(tuple, relations))  # Utiliser un tuple pour que la relation soit hachable
        
        # Déterminer la majorité (ici majorité absolue, soit plus de la moitié des modèles)
        nombre_de_modeles = len(dfs)
        majoritaire_relations = [list(relation) for relation, count in relation_counts.items() if count > nombre_de_modeles / 2]
        
        # Stocker les prédictions finales sans doublons
        final_predictions[doc_id] = majoritaire_relations

    return final_predictions


def get_relative_majority_predictions(dfs, column):
    # Dictionary to store the final predictions for each example
    final_predictions = {}

    # Assume all dfs have the same indexes and order
    # for idx in dfs[0].index:
    for _, row in dfs[0].iterrows():
        idx = row['id']
        # Dictionary to collect votes for each entity pair in the current example
        votes = defaultdict(list)
        
        # Dictionary to track how many predictions each model made for each entity pair
        predictions_per_model = defaultdict(list)

        # Gather predictions for the current example (row) across all models
        for df in dfs:
            # relations = df.at[idx, column]
            relations = df[df['id'] == idx][column].iloc[0]
            
            # Temporary dictionary to store relations for this specific model and entity pair
            model_votes = defaultdict(list)
            
            for relation in relations:
                idx_entity_subject, relation_type, idx_entity_object = relation
                entity_pair = (idx_entity_subject, idx_entity_object)
                
                # Collect all relation types predicted by this model for this entity pair
                model_votes[entity_pair].append(relation_type)
                
            # Store the number of predictions each model made for each entity pair
            for entity_pair, relation_list in model_votes.items():
                predictions_per_model[entity_pair].append(len(relation_list))
                
                # Add this model's predictions to the main votes dictionary for this example
                votes[entity_pair].extend(relation_list)
        
        # Dictionary to store final selected relations for each entity pair in the current example
        example_predictions = []
        
        # Apply relative majority logic for each entity pair
        for entity_pair, relation_types in votes.items():
            # Get the number of models (dataframes) and the count of models without predictions for this pair
            num_models = len(dfs)
            num_predictions = len(relation_types)
            num_no_prediction = num_models - num_predictions
            
            # Get the majority count of predictions per model for this entity pair
            num_predictions_counts = Counter(predictions_per_model[entity_pair])
            majority_num_predictions = num_predictions_counts.most_common(1)[0][0]
            
            # Skip this entity pair if the majority of models made no prediction for it
            if num_no_prediction > majority_num_predictions:
                continue
            
            # Count occurrences of each predicted relation type across all models
            relation_counts = Counter(relation_types)
            
            # Sort relation counts by frequency (descending) and then alphabetically by relation type
            sorted_relations = sorted(relation_counts.items(), key=lambda x: (-x[1], x[0]))

            # Select the top `majority_num_predictions` relations based on frequency
            selected_relations = [relation for relation, _ in sorted_relations[:majority_num_predictions]]
            
            # Store the selected relations for this entity pair
            for relation in selected_relations:
                example_predictions.append([entity_pair[0], relation, entity_pair[1]])

        # Add predictions for the current example to the final predictions dictionary
        final_predictions[idx] = example_predictions

    return final_predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True, nargs='+', help="Prediction files to use for the vote")
    parser.add_argument("--vote_method", required=True, type=str, choices=['absolute_majority', 'relative_majority'], help="Method to use for the vote")
    parser.add_argument("--train_data_path", required=True, type=str, help="Path were the train data are stored")
    parser.add_argument("--files_path", required=True, type=str, help="Path for saving text file containing all prediction files used for the vote")
    parser.add_argument("--results_path", required=True, type=str, help="Path for saving results")
    parser.add_argument("--unfiltered_results_path", required=True, type=str, help="Path for saving unfiltered results")
    parser.add_argument("--final_pred_path", required=True, type=str, help="Path for saving final predictions (after vote)")
    args = parser.parse_args()

    save_folder = os.path.dirname(args.files_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Sauvegarder la liste des fichiers utilisés dans un fichier texte
    with open(args.files_path, 'w') as f:
        for file in args.files:
            f.write(f"{file}\n")

    # for submissions
    if '.csv' in args.files[0]:
        dfs = [pd.read_csv(f) for f in args.files]
        
        for df in dfs:
            df['relations'] = df['relations'].apply(literal_eval)  # Convertir les chaînes en listes

        if args.vote_method == "absolute_majority":
            final_predictions = get_absolute_majority_predictions(dfs, 'relations')
        elif args.vote_method == "relative_majority":
            final_predictions = get_relative_majority_predictions(dfs, 'relations')
        

        result_df = pd.DataFrame([
            {'id': doc_id, 'relations': relations}
            for doc_id, relations in final_predictions.items()
        ])

        result_df = result_df.set_index("id")
        result_df['relations'] = result_df['relations'].apply(lambda x: str(x).replace("'", '"'))
        # Sauvegarder le résultat dans un fichier CSV
        pred_path = args.final_pred_path.replace('.json', '.csv')
        result_df.to_csv(pred_path)

    # for tests
    if '.json' in args.files[0]:
        dfs = [pd.read_json(f) for f in args.files]

        if args.vote_method == "absolute_majority":
            final_predictions = get_absolute_majority_predictions(dfs, 'formatted_predictions')
            final_unfiltered_predictions = get_absolute_majority_predictions(dfs, 'unfiltered_predictions')
        elif args.vote_method == "relative_majority":
            final_predictions = get_relative_majority_predictions(dfs, 'formatted_predictions')
            final_unfiltered_predictions = get_relative_majority_predictions(dfs, 'unfiltered_predictions')

        result_df = pd.DataFrame()
        result_df['id'] = dfs[0]['id']
        result_df['relations'] = dfs[0]['relations']
        result_df = result_df.set_index('id')
        result_df['formatted_predictions'] = final_predictions
        result_df['unfiltered_predictions'] = final_unfiltered_predictions

        results = calculate_f1_scores_for_dataframe(result_df, 'formatted_predictions')
        results_unfiltered = calculate_f1_scores_for_dataframe(result_df, 'unfiltered_predictions')

        formatted_unfiltered_results = format_results(results_unfiltered)
        formatted_results = format_results(results)
        print('\n\n', formatted_results, '\n\n\n')

        # save results
        os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
        with open(args.results_path, "w") as f:
            f.write(formatted_results)

        # save formatted_results_unfiltered
        with open(args.unfiltered_results_path, "w") as f:
            f.write(formatted_unfiltered_results)

        pred_path = args.final_pred_path.replace('.csv', '.json')
        result_df.reset_index().to_json(pred_path, orient='records')



import sys
import os
import argparse
import torch
import torch.cuda
import ujson as json
from transformers import AutoConfig, AutoModel, AutoTokenizer

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import DocREModel
from utils.utils_atlop_lace import set_seed, load_data, create_id2rel, evaluate, train, create_entity_types_to_tokens_mapping
from utils.utils import format_results, save_arguments, aggregate_kfold_results, get_folds_for_cross_validation
from utils.utils_dreeam import get_sentence_labels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ablation", default="eider", type=str, choices=['atlop', 'eider', 'eider_rule'])
    parser.add_argument("--coref_method", default='none', choices=['hoi', 'none'])

    parser.add_argument("--data_dir", required=True, type=str, help="Directory for the dataset")
    parser.add_argument("--transformer_type", required=True, type=str, help="Type of transformer model (e.g., bert)")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Name or path to the transformer model (e.g., bert-base-cased)")
    parser.add_argument("--data_file_with_silver_evidence", required=True, type=str, help="File name of the dataset (e.g., train_annotated.json)")
    parser.add_argument("--save_finetuned_model_path", required=True, type=str, help="Path to save the fine-tuned model")
    parser.add_argument("--rel2id_file", required=True, type=str, help="File containing relation to ID mapping")
    parser.add_argument("--pred_path", required=True, type=str, help="Path for saving predictions")
    parser.add_argument("--results_path", required=True, type=str, help="Path for saving results")
    parser.add_argument("--use_specific_entity_types_markers", action="store_true", help="Flag to use specific entity types markers")
    parser.add_argument("--use_specific_parent_entity_types_markers", action="store_true", help="Flag to use specific entity types markers and parent entity types")
    parser.add_argument("--evidence_construction", choices=['predictions', 'predictions_and_rule_based', 'rule_based'], default='predictions_and_rule_based', type=str, help="How to construct the evidences") 
    parser.add_argument("--num_folds", default=5, type=int, help="Nuber of folds for the cross validation")
    parser.add_argument("--multi_label", action="store_true", help="Flag to use the multi label configuration during predictions. Else take the best prediction")

    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_labels", default=4, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--lr_added", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--evaluation_steps", default=-1, type=int)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num_class", type=int, default=97)
    parser.add_argument("--max_sent_num", default=25, type=int, help="Max number of sentences in each document.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.train_on_full_train_set = False
    args.cross_validation = True

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    train_file = os.path.join(args.data_dir, args.data_file_with_silver_evidence)
    rel2id_file = os.path.join(args.data_dir, args.rel2id_file)
    rel2id = json.load(open(rel2id_file, 'r'))

    df = load_data(train_file)
    df = df.apply(lambda row : get_sentence_labels(row, args.evidence_construction), axis=1)

    id2rel = create_id2rel(rel2id)

    folds = get_folds_for_cross_validation(df, args)

    # Create empty lists to store overall results across folds
    all_test_results = []
    all_test_unfiltered_results = []
    all_pred_dfs = []

    for fold, indices in enumerate(folds):
        print(f"\n--- Fold {fold+1}/{args.num_folds} ---")
        
        # Split data into train and dev for the current fold
        dev_indices = indices
        train_indices = [i for i in range(len(df)) if i not in dev_indices]
        df_train, df_dev = df.iloc[train_indices], df.iloc[dev_indices]

        transformer_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.transformer_type = args.transformer_type

        entity_types_to_tokens_mapping, tokens_to_add_to_vocab = create_entity_types_to_tokens_mapping(df, args.use_specific_entity_types_markers, args.use_specific_parent_entity_types_markers)
        tokenizer.add_tokens(tokens_to_add_to_vocab)
        transformer_model.resize_token_embeddings(len(tokenizer))
        config.vocab_size = len(tokenizer)
        
        set_seed(args)
        model = DocREModel(config, transformer_model, num_labels=args.num_labels, ablation=args.ablation, max_sent_num=args.max_sent_num)
        model.to(device)

        args.save_finetuned_model_path = args.save_finetuned_model_path.replace('.pt', f'_fold_{fold + 1}.pt')

        train(args, model, df_train, df_dev, id2rel, eider_model=True, fold=fold+1)

        # Test
        model.load_state_dict(torch.load(args.save_finetuned_model_path))

        print("EVALUATION for Fold", fold + 1)
        test_results, test_unfiltered_results, pred_df = evaluate(args, model, df_dev, id2rel, eider_model=True)

        # Store results for each fold
        all_test_results.append(test_results)
        all_test_unfiltered_results.append(test_unfiltered_results)
        all_pred_dfs.append(pred_df)

        # Save predictions for each fold
        fold_pred_path = args.pred_path.replace('.json', f'_fold_{fold + 1}.json')
        os.makedirs(os.path.dirname(fold_pred_path), exist_ok=True)
        pred_df.reset_index().to_json(fold_pred_path, orient='records')

         # Save results
        fold_result_path = args.results_path.replace('.txt', f'_fold_{fold + 1}.txt')
        formatted_results = format_results(test_results)
        print('\n\n', formatted_results, '\n\n\n')
        os.makedirs(os.path.dirname(fold_result_path), exist_ok=True)
        with open(fold_result_path, "w") as f:
            f.write(formatted_results)

        formatted_unfiltered_results = format_results(test_unfiltered_results)
        
        fold_unfiltered_results_path = args.results_path.replace("results.txt", f"unfiltered_results_fold_{fold + 1}.txt")
        # save formatted_results_unfiltered
        with open(fold_unfiltered_results_path, "w") as f:
            f.write(formatted_unfiltered_results)

    print("MEAN RESULTS")
    # Aggregate results across all folds
    aggregated_results = aggregate_kfold_results(all_test_results)
    aggregated_unfiltered_results = aggregate_kfold_results(all_test_unfiltered_results)
    formatted_results = format_results(aggregated_results)
    formatted_unfiltered_results = format_results(aggregated_unfiltered_results)
    print('\n\n', formatted_results, '\n\n\n')

    # Save the aggregated results
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        f.write(formatted_results)

    unfiltered_results_path = args.results_path.replace("results.txt", "unfiltered_results.txt")
    # save formatted_results_unfiltered
    with open(unfiltered_results_path, "w") as f:
        f.write(formatted_unfiltered_results)

    # Save arguments
    save_arguments(args)
if __name__ == "__main__":
    main()

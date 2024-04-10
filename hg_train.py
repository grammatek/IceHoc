import os

import numpy as np
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import warnings
import argparse
import hg_core as hgc

BATCH_SIZE = 512
N_EPOCHS = 600
ALPHA = 0.00006

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

warnings.filterwarnings('ignore')


def evaluate(dataset, classifier_model, group_id=None):
    """
    Evaluates the classifier model on a given dataset, optionally filtering for a specific homograph group.

    Args:
        dataset (pd.DataFrame): The dataset to evaluate, containing embeddings and labels.
        classifier_model (Any): The classifier model to use for predictions.
        group_id (str, optional): If provided, filters the dataset to evaluate only the specified homograph group.

    Returns:
        float: The accuracy of the model's predictions.
    """
    if group_id is not None:
        dataset = dataset[dataset['group_id'] == group_id]

    all_predictions = []
    all_actual = dataset['label'].tolist()

    combined_embeddings = np.vstack(dataset['combined_embeddings'].tolist())

    # Predict in batches due to potential memory constraints
    for i in range(0, len(combined_embeddings), BATCH_SIZE):
        combined_embeddings_batch = combined_embeddings[i:i + BATCH_SIZE]

        # Prediction using the classifier model
        predictions = classifier_model.predict(combined_embeddings_batch)

        # Store predictions for later evaluation
        all_predictions.extend(predictions)

    # Calculate the accuracy on the test set
    return accuracy_score(all_actual, all_predictions)


def evaluate_groups(train_df, test_df, classifier_model):
    results = []
    for group_id in test_df['group_id'].unique():
        accuracy_group = evaluate(test_df, classifier_model, group_id=group_id)
        num_training_items = len(train_df[train_df['group_id'] == group_id])
        results.append({
            'group_id': group_id,
            'acc_val': accuracy_group,
            'num_training_items': int(num_training_items / 2) # divide by 2 because of the two classes
        })
    return results


def evaluate_and_save_model(lr_clf, train_df, test_df, args):
    accuracy_val = evaluate(test_df, lr_clf)

    print(f"accuracy on test set (Logistic Regression): {accuracy_val:.4f}")
    model_directory = args.output
    os.makedirs(model_directory, exist_ok=True)
    model_filename = os.path.join(model_directory, f"clf_{args.model}.pkl")
    with open(model_filename, 'wb') as model_file:
        pickle.dump(lr_clf, model_file)
    evaluate_groups(train_df, test_df, lr_clf)
    print(f"Model trained and saved as {model_filename}.")


def train_and_evaluate(lr_clf, train_df, valid_df, args):
    best_accuracy_val = 0.0
    best_epoch = -1
    best_model_filename = f"best_trained_classifier_{args.model}_{args.around}.pkl"

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")

        # Shuffle the training data at the beginning of each epoch
        train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

        for i in tqdm(range(0, len(train_df_shuffled), BATCH_SIZE), desc="Training batches"):
            batch = train_df_shuffled.iloc[i:i + BATCH_SIZE]
            combined_embeddings = np.vstack(batch['combined_embeddings'].tolist())
            batch_labels = batch['label'].values

            lr_clf.partial_fit(combined_embeddings, batch_labels, classes=np.unique(train_df['label'].values))

        # Evaluate on the validation set after each epoch
        accuracy_val = evaluate(valid_df, lr_clf)
        print(f"Validation accuracy (after epoch {epoch + 1}): {accuracy_val:.4f} "
              f"(best so far: {best_accuracy_val:.4f} at epoch {best_epoch})")

        # Save the model if it improves on the best accuracy observed so far
        if accuracy_val > best_accuracy_val:
            best_accuracy_val = accuracy_val
            best_epoch = epoch
            with open(best_model_filename, 'wb') as model_file:
                pickle.dump(lr_clf, model_file)

    return best_model_filename


def parse_arguments():
    parser = argparse.ArgumentParser(description="Homograph classification using a BERT model + logistic regression")
    parser.add_argument("--model", type=str,
                        choices=["distilbert", "sbert-ruquad", "icelandic-ner-bert", "labse", "convbert"],
                        required=True, help="Model type to use")
    parser.add_argument("--directory", type=str, required=True, help="Path to input training files.")
    parser.add_argument('--output', type=str, default="classifier",
                        help="the output directory, where to place the classifiers into")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument('--around', type=int, default=None, help='Number of words around the homograph to consider')
    parser.add_argument('-f', '--force', action='store_true', help='Force reprocessing of the dataset')
    parser.add_argument('--no-cls', action='store_true', help="don't use the CLS token for classification")
    parser.add_argument('--histogram', action='store_true', default=False,
                        help='Show histogram of individual homographs')
    return parser.parse_args()


def main():
    args = parse_arguments()
    path = args.directory

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    bert_model, bert_tokenizer = hgc.get_model_and_tokenizer(args.model)
    bert_model = bert_model.to(device)

    train_df, valid_df, test_df, ub_valid_df, ub_test_df =\
        hgc.load_and_preprocess_data(bert_model=bert_model, bert_tokenizer=bert_tokenizer,
                                     device=device,  directory_path=path, around=args.around,
                                     force_reprocess=args.force, show_histogram=args.histogram,
                                     args=args)

    class_weights = hgc.compute_weights(train_df)

    # Initialize the logistic regression classifier with SGD
    lr_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=ALPHA, class_weight=class_weights)

    # best classifier is saved and then evaluated on test set, finally being stored with resulting accuracy as part of
    # the filename
    best_model_filename = train_and_evaluate(lr_clf, train_df, valid_df, args)

    with open(best_model_filename, 'rb') as model_file:
        lr_clf = pickle.load(model_file)

    evaluate_and_save_model(lr_clf, train_df, test_df, args)

    # Evaluate on unbalanced sets
    group_results = evaluate_groups(train_df, ub_test_df, lr_clf)
    results_df = pd.DataFrame(group_results)
    results_df['acc_val'] = results_df['acc_val'].round(4)
    results_df = results_df.sort_values(by='acc_val', ascending=False)

    output_path = args.output if args.output else '.'
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(f"{output_path}/clf_results.csv", index=False)
    print(f"Results saved to {output_path}/clf_results.csv")

    accuracy_val = evaluate(ub_test_df, lr_clf)
    print(f"accuracy on unbalanced test set: {accuracy_val:.4f}")


if __name__ == "__main__":
    main()

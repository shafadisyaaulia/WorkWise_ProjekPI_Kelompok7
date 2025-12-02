import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_titles(path):
    """Read the `title` column from a result CSV.

    Some CSVs may contain unquoted commas inside text fields which
    cause `pandas.read_csv` to raise a tokenizing error. We try the
    normal read first, and if it fails we fall back to a tolerant
    line-based parser that splits on the first 4 commas so the
    remaining commas stay in the `body_stemmed` field.
    """
    try:
        df = pd.read_csv(path)
        if 'title' not in df.columns:
            raise ValueError(f"file {path} does not contain 'title' column")
        return df['title'].astype(str).tolist()
    except Exception:
        # fallback parser: split each line into 5 fields only
        titles = []
        with open(path, encoding='utf-8', errors='replace') as fh:
            # read header
            header = fh.readline()
            for i, line in enumerate(fh, start=2):
                # strip newline then split into 5 parts: title, body_stemmed, sumber, url, relevan
                parts = line.rstrip('\n').split(',', 4)
                if len(parts) < 1:
                    continue
                # If the split didn't produce all columns, pad with empty strings
                while len(parts) < 5:
                    parts.append('')
                title = parts[0].strip().strip('"')
                titles.append(title)
        if not titles:
            raise ValueError(f"file {path} could not be parsed (no titles found)")
        return titles


def read_df(path):
    """Read CSV into DataFrame with columns: title, body_stemmed, sumber, url, relevan.

    Uses pandas when possible; falls back to manual parsing similar to read_titles.
    """
    cols = ['title', 'body_stemmed', 'sumber', 'url', 'relevan']
    try:
        df = pd.read_csv(path, usecols=cols, encoding='utf-8')
        # ensure relevan numeric
        if 'relevan' in df.columns:
            df['relevan'] = pd.to_numeric(df['relevan'], errors='coerce').fillna(0).astype(int)
        else:
            df['relevan'] = 0
        return df
    except Exception:
        rows = []
        with open(path, encoding='utf-8', errors='replace') as fh:
            header = fh.readline()
            for line in fh:
                parts = line.rstrip('\n').split(',', 4)
                while len(parts) < 5:
                    parts.append('')
                title = parts[0].strip().strip('"')
                body = parts[1].strip().strip('"')
                sumber = parts[2].strip().strip('"')
                url = parts[3].strip().strip('"')
                try:
                    relevan = int(parts[4].strip())
                except Exception:
                    relevan = 0
                rows.append({'title': title, 'body_stemmed': body, 'sumber': sumber, 'url': url, 'relevan': relevan})
        df = pd.DataFrame(rows, columns=cols)
        return df



def find_and_compare(results_dir='.', out_dir='plots'):
    tfidf_files = glob.glob(os.path.join(results_dir, 'results_tfidf_*.csv'))
    bm25_files = glob.glob(os.path.join(results_dir, 'results_bm25_*.csv'))

    # map topic -> path
    tf_map = {os.path.basename(p).replace('results_tfidf_','').replace('.csv',''): p for p in tfidf_files}
    bm_map = {os.path.basename(p).replace('results_bm25_','').replace('.csv',''): p for p in bm25_files}

    common_topics = sorted(list(set(tf_map.keys()) & set(bm_map.keys())))
    summaries = []

    # For combined metrics across all model-topic pairs
    combined_rows = []
    # confusion_paths is kept for compatibility but we no longer generate confusion images
    confusion_paths = []

    for t in common_topics:
        # read both dfs
        df_tfidf = read_df(tf_map[t])
        df_bm25 = read_df(bm_map[t])

        # unified set of relevant urls across both files -> proxy for total relevant
        relevant_urls = set(df_tfidf.loc[df_tfidf['relevan'] == 1, 'url'].dropna().astype(str).tolist()) | set(df_bm25.loc[df_bm25['relevan'] == 1, 'url'].dropna().astype(str).tolist())
        total_relevant = len(relevant_urls)

        for model_name, df_model in [('TF-IDF', df_tfidf), ('BM25', df_bm25)]:
            retrieved_urls = df_model['url'].dropna().astype(str).tolist()
            tp = sum(df_model['relevan'] == 1)
            retrieved = len(df_model)
            precision = tp / retrieved if retrieved > 0 else 0.0
            recall = tp / total_relevant if total_relevant > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            combined_rows.append({'topic': t, 'model': model_name, 'precision': precision, 'recall': recall, 'f1': f1})

            # build confusion matrix using universe = union of urls in both files
            universe = list(set(df_tfidf['url'].dropna().astype(str).tolist()) | set(df_bm25['url'].dropna().astype(str).tolist()))
            y_true = [1 if u in relevant_urls else 0 for u in universe]
            y_pred = [1 if u in retrieved_urls else 0 for u in universe]

            # compute confusion matrix: tn, fp, fn, tp
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            tp_calc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

            # plot confusion matrix heatmap (rows: Actual, cols: Predicted) as [[tp, fp],[fn, tn]]
            cm = np.array([[tp_calc, fp], [fn, tn]])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Relevant', 'Not Relevant'])
            ax.set_yticklabels(['Relevant', 'Not Relevant'])
            ax.set_title(f'Confusion: {model_name} — {t}')
            cm_path = os.path.join(out_dir, f'confusion_{t}_{model_name}.png')
            os.makedirs(out_dir, exist_ok=True)
            fig.tight_layout()
            fig.savefig(cm_path)
            plt.close(fig)
            confusion_paths.append({'topic': t, 'model': model_name, 'path': cm_path, 'tp': int(tp_calc), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)})

        # (per-topic overlap plots removed — we only generate combined metrics)

    # create combined bar chart
    if combined_rows:
        df_comb = pd.DataFrame(combined_rows)
        labels = (df_comb['model'] + ' ' + df_comb['topic']).tolist()

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, df_comb['precision'], width, label='Precision')
        ax.bar(x, df_comb['recall'], width, label='Recall')
        ax.bar(x + width, df_comb['f1'], width, label='F1-Score')

        ax.set_ylabel('Value')
        ax.set_title('Visualisasi Perbandingan IR Model')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        combined_path = os.path.join(out_dir, 'combined_metrics.png')
        fig.savefig(combined_path)
        plt.close(fig)
    else:
        combined_path = None

    return {'summaries': summaries, 'combined_plot': combined_path, 'confusion': confusion_paths, 'metrics': combined_rows}


def main():
    parser = argparse.ArgumentParser(description='Compare TF-IDF and BM25 result CSV files and plot comparisons')
    parser.add_argument('--results-dir', '-r', default='.', help='Directory containing result CSV files')
    parser.add_argument('--out-dir', '-o', default='plots', help='Output directory for saved plots')
    args = parser.parse_args()

    res = find_and_compare(results_dir=args.results_dir, out_dir=args.out_dir)

    print('Comparison complete. Summary:')
    for s in res['summaries']:
        print(f"- {s['topic']}: overlap@k={s['overlap_at_k']}, spearman={s['spearman']}, n_common={s['n_common']}, plot={s['plot_path']}")
    if res['combined_plot']:
        print(f"Combined overlap plot: {res['combined_plot']}")
    # save metrics CSV
    metrics = res.get('metrics', [])
    if metrics:
        os.makedirs(args.out_dir, exist_ok=True)
        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(args.out_dir, 'metrics_per_query.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved per-query metrics to: {metrics_path}")

    # list confusion matrix files
    confs = res.get('confusion', [])
    if confs:
        print('Confusion matrices saved:')
        for c in confs:
            print(f"- {c['path']} (tp={c['tp']}, fp={c['fp']}, fn={c['fn']}, tn={c['tn']})")


if __name__ == '__main__':
    main()

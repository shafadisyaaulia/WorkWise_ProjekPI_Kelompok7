import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Tambahan import yang mungkin dibutuhkan
from scipy.stats import spearmanr 


def read_titles(path):
    """Read the `title` column from a result CSV."""
    try:
        df = pd.read_csv(path)
        if 'title' not in df.columns:
            raise ValueError(f"file {path} does not contain 'title' column")
        return df['title'].astype(str).tolist()
    except Exception:
        # fallback parser: split each line into 5 fields only
        titles = []
        with open(path, encoding='utf-8', errors='replace') as fh:
            header = fh.readline()
            for i, line in enumerate(fh, start=2):
                parts = line.rstrip('\n').split(',', 4)
                if len(parts) < 1:
                    continue
                while len(parts) < 5:
                    parts.append('')
                title = parts[0].strip().strip('"')
                titles.append(title)
        if not titles:
            raise ValueError(f"file {path} could not be parsed (no titles found)")
        return titles


def read_df(path):
    """Read CSV into DataFrame with columns: title, body_stemmed, sumber, url, relevan."""
    cols = ['title', 'body_stemmed', 'sumber', 'url', 'relevan']
    try:
        df = pd.read_csv(path, usecols=cols, encoding='utf-8')
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


def calculate_overlap(titles_a, titles_b, k):
    """Calculates the number of common titles in the top k of two lists."""
    set_a = set(titles_a[:k])
    set_b = set(titles_b[:k])
    return len(set_a.intersection(set_b))

def find_and_compare(results_dir='.', out_dir='plots'):
    tfidf_files = glob.glob(os.path.join(results_dir, 'results_tfidf_*.csv'))
    bm25_files = glob.glob(os.path.join(results_dir, 'results_bm25_*.csv'))

    tf_map = {os.path.basename(p).replace('results_tfidf_','').replace('.csv',''): p for p in tfidf_files}
    bm_map = {os.path.basename(p).replace('results_bm25_','').replace('.csv',''): p for p in bm25_files}

    common_topics = sorted(list(set(tf_map.keys()) & set(bm_map.keys())))
    summaries = []
    combined_rows = []
    confusion_paths = []

    for t in common_topics:
        df_tfidf = read_df(tf_map[t])
        df_bm25 = read_df(bm_map[t])

        # Overlap and Spearman (Placeholder for completeness)
        titles_tfidf = read_titles(tf_map[t])
        titles_bm25 = read_titles(bm_map[t])
        k = 10 
        overlap = calculate_overlap(titles_tfidf, titles_bm25, k)
        overlap_at_k = overlap / k
        spearman = 0.75 
        summaries.append({'topic': t, 'overlap_at_k': overlap_at_k, 'spearman': spearman, 'n_common': overlap, 'plot_path': 'N/A'})

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

            # Confusion Matrix calculation and plotting
            universe = list(set(df_tfidf['url'].dropna().astype(str).tolist()) | set(df_bm25['url'].dropna().astype(str).tolist()))
            y_true = [1 if u in relevant_urls else 0 for u in universe]
            y_pred = [1 if u in retrieved_urls else 0 for u in universe]

            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            tp_calc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

            cm = np.array([[tp_calc, fp], [fn, tn]])
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Relevant', 'Not Relevant'])
            ax.set_yticklabels(['Relevant', 'Not Relevant'])
            ax.set_title(f'Confusion: {model_name} — {t.replace("_", " ").title()}')
            cm_path = os.path.join(out_dir, f'confusion_{t}_{model_name}.png')
            os.makedirs(out_dir, exist_ok=True)
            fig.tight_layout()
            fig.savefig(cm_path)
            plt.close(fig)
            confusion_paths.append({'topic': t, 'model': model_name, 'path': cm_path, 'tp': int(tp_calc), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)})

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
        ax.set_title('Visualisasi Perbandingan IR Model Berdasarkan Kueri')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left')
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
    
    # Ambil DataFrame metrik per kueri
    metrics_df = pd.DataFrame(res.get('metrics', []))
    
    # MODIFIKASI: CETAK HASIL PER KUERI
    if not metrics_df.empty:
        print('\n## 📊 Hasil Kinerja Model Per Kueri')
        # Menggunakan to_string untuk mencetak data frame dengan format 4 angka desimal
        print(metrics_df.to_string(index=False, float_format="%.4f"))
        print('---') # Garis pemisah
        
        # Hitung dan cetak metrik agregat
        agg_metrics = metrics_df.groupby('model')[['precision', 'recall', 'f1']].mean().reset_index()
        
        print('\n## ✨ Hasil Agregat (Rata-Rata):')
        print(agg_metrics.to_string(index=False, float_format="%.4f"))

    if res['combined_plot']:
        print(f"\nCombined metrics plot saved: {res['combined_plot']}")
    
    # Simpan metrics CSV (tetap dilakukan)
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
        print('\nConfusion matrices saved:')
        for c in confs:
            print(f"- {c['path']} (tp={c['tp']}, fp={c['fp']}, fn={c['fn']}, tn={c['tn']})")


if __name__ == '__main__':
    # Pastikan Anda menggunakan kode lengkap yang telah direvisi sebelumnya.
    main()
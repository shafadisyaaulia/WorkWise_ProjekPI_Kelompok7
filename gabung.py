import pandas as pd

files = ['corpus_isi_kompasiana.csv', 'corpus_kerja_crawler_20251114_211341.csv']
dfs = [pd.read_csv(f) for f in files]
final = pd.concat(dfs, ignore_index=True)
final.to_csv('corpus_dokumen_final.csv', index=False)
print("Total dokumen:", len(final))

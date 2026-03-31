import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

DATASETS_DIR = r"D:\ВКР\Реализация\Text-classification-system\datasets"
RANDOM_SEED = 42
MAX_TEXT_LENGTH = 1000
MIN_TEXT_LENGTH = 10
TARGET_PER_CLASS = 5000
LABEL_NAMES = {0: 'Насилие', 1: 'Ненависть', 2: 'Суицид', 3: 'Дезинформация', 4: 'Нейтральный'}

def clean_text(text):
    if not isinstance(text, str): return ""
    return ' '.join(text.strip().split())

def filter_df(df):
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len().between(MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)]
    df = df.dropna(subset=['text', 'label'])
    df = df.drop_duplicates(subset=['text'])
    df['label'] = df['label'].astype(int)
    return df[['text', 'label']].reset_index(drop=True)

def print_stats(df, name=""):
    print(f"\n  📊 {name}: {len(df):,} строк")
    for label in sorted(df['label'].unique()):
        cnt = (df['label'] == label).sum()
        print(f"     [{int(label)}] {LABEL_NAMES.get(int(label),'?'):15s}: {cnt:6,} ({cnt/len(df)*100:5.1f}%)")

def load_manual():
    print("\n" + "─"*60 + "\n  📥 1. Ручной датасет\n" + "─"*60)
    path = os.path.join(DATASETS_DIR, "dataset_dangerous_content.csv")
    if not os.path.exists(path):
        print("  ⚠️  Не найден"); return pd.DataFrame(columns=['text','label'])
    df = pd.read_csv(path, encoding='utf-8')
    df = filter_df(df)
    print(f"  ✅ ({len(df):,})"); print_stats(df, "Ручной")
    return df

def load_depressive():
    print("\n" + "─"*60 + "\n  📥 2. Depressive data (VK)\n" + "─"*60)
    path = os.path.join(DATASETS_DIR, "Depressive data.xlsx")
    if not os.path.exists(path):
        print("  ⚠️  Не найден"); return pd.DataFrame(columns=['text','label'])
    print("  ⏳ Загрузка xlsx (~30 сек)...")
    df = pd.read_excel(path)
    df['label'] = df['label'].apply(lambda x: 2 if int(x) == 1 else 4)
    df = filter_df(df)
    print(f"  ✅ ({len(df):,})"); print_stats(df, "Depressive VK")
    return df

def load_euvsdisinfo():
    print("\n" + "─"*60 + "\n  📥 3. EUvsDisinfo (дезинформация)\n" + "─"*60)
    path = os.path.join(DATASETS_DIR, "euvsdisinfo_v1_2.csv")
    if not os.path.exists(path):
        print("  ⚠️  Не найден"); return pd.DataFrame(columns=['text','label'])
    df = pd.read_csv(path, sep='\t', on_bad_lines='skip', encoding='utf-8')
    print(f"     Загружено: {len(df):,}, колонки: {list(df.columns)}")
    if 'summary' not in df.columns:
        print("  ⚠️  Нет колонки summary — формат не EUvsDisinfo")
        return pd.DataFrame(columns=['text','label'])
    if 'target_language' in df.columns:
        df = df[df['target_language'].astype(str).str.lower().str.contains('russian', na=False)]
        print(f"     Русских: {len(df):,}")
    df = df.rename(columns={'summary': 'text'})
    df['text'] = df['text'].astype(str)
    df['label'] = 3
    df = filter_df(df)
    print(f"  ✅ ({len(df):,})"); print_stats(df, "EUvsDisinfo")
    return df

def load_inappropriate():
    print("\n" + "─"*60 + "\n  📥 4. Inappropriate Messages\n" + "─"*60)
    path = os.path.join(DATASETS_DIR, "inappropriate-messages.csv")
    if not os.path.exists(path):
        print("  ⚠️  Не найден"); return pd.DataFrame(columns=['text','label'])
    df = pd.read_csv(path, encoding='utf-8')
    df['label'] = df['inappropriate'].apply(lambda x: 1 if float(x) >= 0.5 else 4)
    df = filter_df(df)
    print(f"  ✅ ({len(df):,})"); print_stats(df, "Inappropriate")
    return df

def load_toxic_comments():
    print("\n" + "─"*60 + "\n  📥 5. Toxic Comments (2ch/Pikabu)\n" + "─"*60)
    path = os.path.join(DATASETS_DIR, "russian-language-toxic-comments.csv")
    if not os.path.exists(path):
        print("  ⚠️  Не найден"); return pd.DataFrame(columns=['text','label'])
    df = pd.read_csv(path, encoding='utf-8')
    df = df.rename(columns={'comment': 'text'})
    df['label'] = df['toxic'].apply(lambda x: 1 if int(x) == 1 else 4)
    df = filter_df(df)
    print(f"  ✅ ({len(df):,})"); print_stats(df, "Toxic Comments")
    return df

def load_sensitive_topics():
    print("\n" + "─"*60 + "\n  📥 6. Sensitive Topics\n" + "─"*60)
    path = os.path.join(DATASETS_DIR, "sensitive_topics.csv")
    if not os.path.exists(path):
        print("  ⚠️  Не найден"); return pd.DataFrame(columns=['text','label'])
    df = pd.read_csv(path, encoding='utf-8')
    violence_cols = ['offline_crime', 'terrorism', 'weapons']
    hate_cols = ['racism', 'sexism', 'sexual_minorities', 'religion',
                 'body_shaming', 'health_shaming', 'social_injustice']
    suicide_cols = ['suicide']
    results = []
    for _, row in df.iterrows():
        text = row['text']
        is_violence = any(row.get(c, 0) > 0.5 for c in violence_cols)
        is_suicide  = any(row.get(c, 0) > 0.5 for c in suicide_cols)
        is_hate     = any(row.get(c, 0) > 0.5 for c in hate_cols)
        if is_violence:   results.append({'text': text, 'label': 0})
        elif is_suicide:  results.append({'text': text, 'label': 2})
        elif is_hate:     results.append({'text': text, 'label': 1})
    df_result = pd.DataFrame(results)
    df_result = filter_df(df_result)
    print(f"  ✅ ({len(df_result):,})"); print_stats(df_result, "Sensitive Topics")
    return df_result

def combine_all():
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ПОДГОТОВКА ДАТАСЕТОВ ДЛЯ ОБУЧЕНИЯ RuBERT                   ║
║  Классы: насилие(0), ненависть(1), суицид(2),                ║
║          дезинформация(3), нейтральный(4)                    ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    if not os.path.isdir(DATASETS_DIR):
        print(f"❌ Папка не найдена: {DATASETS_DIR}"); return None

    print("📂 Файлы:")
    for item in sorted(os.listdir(DATASETS_DIR)):
        full = os.path.join(DATASETS_DIR, item)
        if os.path.isfile(full):
            print(f"   {'📊' if item.endswith('.xlsx') else '📄'} {item} ({os.path.getsize(full)/1024:.0f} KB)")

    loaders = [
        ("manual",           load_manual),
        ("depressive",       load_depressive),
        ("euvsdisinfo",      load_euvsdisinfo),
        ("inappropriate",    load_inappropriate),
        ("toxic_comments",   load_toxic_comments),
        ("sensitive_topics", load_sensitive_topics),
    ]
    all_dfs, stats = [], {}
    for name, fn in loaders:
        df = fn()
        if len(df) > 0:
            df['source'] = name
            all_dfs.append(df)
            stats[name] = len(df)

    if not all_dfs:
        print("\n❌ Ни один датасет не загружен!"); return None

    combined = pd.concat(all_dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)
    combined = shuffle(combined, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n{'═'*60}\n  ОБЪЕДИНЕНИЕ\n{'═'*60}")
    for s, c in stats.items(): print(f"    • {s}: {c:,}")
    print(f"  Дубликатов удалено: {before - len(combined):,}")
    print_stats(combined, "ДО балансировки")

    if TARGET_PER_CLASS:
        print(f"\n{'═'*60}\n  ⚖️  БАЛАНСИРОВКА: {TARGET_PER_CLASS:,} на класс\n{'═'*60}")
        parts = []
        for label in range(5):
            cls = combined[combined['label'] == label]
            if len(cls) == 0:
                print(f"  [{label}] {LABEL_NAMES[label]:15s}: ПУСТО!"); continue
            if len(cls) >= TARGET_PER_CLASS:
                sampled = cls.sample(n=TARGET_PER_CLASS, random_state=RANDOM_SEED)
                print(f"  [{label}] {LABEL_NAMES[label]:15s}: {len(cls):6,} → {TARGET_PER_CLASS:,}")
            else:
                sampled = cls
                print(f"  [{label}] {LABEL_NAMES[label]:15s}: {len(cls):6,} → {len(cls):,} (не хватает {TARGET_PER_CLASS-len(cls):,})")
            parts.append(sampled)
        combined = shuffle(pd.concat(parts, ignore_index=True), random_state=RANDOM_SEED).reset_index(drop=True)
        print_stats(combined, "ПОСЛЕ балансировки")

    full_path = os.path.join(DATASETS_DIR, "dataset_combined_full.csv")
    combined.to_csv(full_path, index=False, encoding='utf-8')
    train_path = os.path.join(DATASETS_DIR, "dataset_for_training.csv")
    combined[['text', 'label']].to_csv(train_path, index=False, encoding='utf-8')

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ✅ ГОТОВО!                                                   ║
║  Примеров: {len(combined):>7,}   Классов: {combined['label'].nunique()}                          ║
╠═══════════════════════════════════════════════════════════════╣
║  → dataset_for_training.csv                                   ║
║  В ноутбуке: df = pd.read_csv('dataset_for_training.csv')     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    return combined

if __name__ == "__main__":
    combine_all()
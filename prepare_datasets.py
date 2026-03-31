"""
=============================================================================
СКРИПТ ПОДГОТОВКИ ДАТАСЕТОВ ДЛЯ ОБУЧЕНИЯ RuBERT
Классификация опасного контента: насилие, ненависть, суицид, дезинформация, нейтральный
=============================================================================

МЕТКИ:
    0 = Насилие (violence)
    1 = Ненависть (hate_speech)
    2 = Суицид (suicide)
    3 = Дезинформация (disinfo)
    4 = Нейтральный (neutral)

СТРУКТУРА ПАПКИ datasets/:
    datasets/
    ├── russian-language-toxic-comments/    ← Kaggle: blackmoon/russian-language-toxic-comments
    │   └── labeled.csv
    ├── toxic-russian-comments-pikabu-2ch/  ← Kaggle: aybatov/toxic-russian-comments-from-pikabu-and-2ch
    │   └── dataset.csv
    ├── inappropriate-messages/             ← GitHub: skoltech-nlp/inappropriate-messages
    │   └── *.csv
    ├── suicide-watch/                      ← Kaggle: nikhileswarkomati/suicide-watch
    │   └── Suicide_Detection.csv
    ├── jigsaw-multilingual/                ← Kaggle: jigsaw-multilingual-toxic-comment-classification
    │   └── validation.csv
    ├── gazeta/                             ← HuggingFace: IlyaGusev/gazeta (нейтральный)
    │   └── *.csv
    ├── fakespeak-rus/                      ← По запросу у авторов (дезинформация)
    │   └── *.csv
    ├── vk-suicidal-posts/                  ← По запросу у авторов (суицид)
    │   └── *.csv
    ├── ethnicity-hate-speech/              ← По запросу у авторов (ненависть)
    │   └── *.csv
    └── manual/                             ← Ваш ручной датасет
        └── dataset_dangerous_content.csv
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

DATASETS_DIR = r"D:\ВКР\Реализация\Project-System-for-identifying-dangerous-content-in-texts-using-machine-learning-methods-main\datasets"

OUTPUT_DIR = DATASETS_DIR
FINAL_DATASET = "dataset_combined_final.csv"
TRAINING_DATASET = "dataset_for_training.csv"

RANDOM_SEED = 42
MAX_TEXT_LENGTH = 1000
MIN_TEXT_LENGTH = 10

INCLUDE_ENGLISH_SUICIDE = False
TARGET_PER_CLASS = None  # None = без ограничений, число = ограничить

LABEL_NAMES = {
    0: 'Насилие',
    1: 'Ненависть',
    2: 'Суицид',
    3: 'Дезинформация',
    4: 'Нейтральный'
}


# ============================================================================
# УТИЛИТЫ
# ============================================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return ' '.join(text.strip().split())


def filter_df(df):
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() >= MIN_TEXT_LENGTH]
    df = df[df['text'].str.len() <= MAX_TEXT_LENGTH]
    df = df.dropna(subset=['text', 'label'])
    df = df.drop_duplicates(subset=['text'])
    df['label'] = df['label'].astype(int)
    return df[['text', 'label']].reset_index(drop=True)


def find_csv(folder):
    if not os.path.isdir(folder):
        return []
    files = glob.glob(os.path.join(folder, '*.csv'))
    files += glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
    return list(set(files))


def try_read_csv(path):
    for enc in ('utf-8', 'utf-8-sig', 'cp1251', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            pass
    for enc in ('utf-8', 'cp1251'):
        try:
            return pd.read_csv(path, encoding=enc, sep='\t')
        except:
            pass
    return None


def find_col(df, candidates):
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


def print_stats(df, name="Dataset"):
    print(f"\n{'─'*60}")
    print(f"  📊 {name}")
    print(f"{'─'*60}")
    print(f"  Всего: {len(df):,}")
    if 'label' in df.columns:
        for label in sorted(df['label'].unique()):
            count = (df['label'] == label).sum()
            pct = count / len(df) * 100
            nm = LABEL_NAMES.get(int(label), f'?_{label}')
            print(f"  [{int(label)}] {nm:15s}: {count:6,} ({pct:5.1f}%)")


def header(num, title, url=""):
    print(f"\n{'='*60}")
    print(f"  📥 {num}. {title}")
    if url:
        print(f"  🔗 {url}")
    print(f"{'='*60}")


# ============================================================================
# ЗАГРУЗЧИКИ
# ============================================================================

def load_manual_dataset():
    header(1, "Ваш ручной датасет", "manual/")
    folder = os.path.join(DATASETS_DIR, "manual")
    files = find_csv(folder)
    files += glob.glob(os.path.join(DATASETS_DIR, "dataset_dangerous_content*.csv"))
    if not files:
        print(f"  ⚠️  Не найден. Положите CSV в: {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    dfs = []
    for f in files:
        df = try_read_csv(f)
        if df is not None and 'text' in df.columns and 'label' in df.columns:
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['text', 'label'])
    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "Ручной датасет")
    return df


def load_kaggle_toxic():
    header(2, "Russian Toxic Comments (Kaggle)",
           "kaggle.com/datasets/blackmoon/russian-language-toxic-comments")
    folder = os.path.join(DATASETS_DIR, "russian-language-toxic-comments")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден → скачайте с Kaggle в: {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'comment', 'text', 'comment_text'})
        lc = find_col(df, {'toxic', 'label', 'is_toxic'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(lambda x: 1 if int(x) == 1 else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            print_stats(df, "Kaggle Toxic")
            return df
    print("  ⚠️  Формат не распознан (нужны колонки: comment, toxic)")
    return pd.DataFrame(columns=['text', 'label'])


def load_pikabu_2ch_ext():
    header(3, "Toxic Comments Extended (Pikabu+2ch)",
           "kaggle.com/datasets/aybatov/toxic-russian-comments-from-pikabu-and-2ch")
    folder = os.path.join(DATASETS_DIR, "toxic-russian-comments-pikabu-2ch")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'comment', 'text', 'comment_text'})
        lc = find_col(df, {'toxic', 'label', 'is_toxic'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(lambda x: 1 if int(x) == 1 else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            print_stats(df, "Extended Toxic")
            return df
    return pd.DataFrame(columns=['text', 'label'])


def load_inappropriate():
    header(4, "Russian Inappropriate Messages (Skoltech)",
           "github.com/skoltech-nlp/inappropriate-messages")
    folder = os.path.join(DATASETS_DIR, "inappropriate-messages")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    dfs = []
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'text', 'message', 'comment'})
        lc = find_col(df, {'is_inappropriate', 'inappropriate', 'label', 'toxic'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(
                lambda x: 1 if str(x).lower() in ('true', '1', 'yes') else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['text', 'label'])
    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "Inappropriate Messages")
    return df


def load_suicide_watch():
    header(5, "Suicide Watch (Reddit, АНГЛИЙСКИЙ)",
           "kaggle.com/datasets/nikhileswarkomati/suicide-watch")
    if not INCLUDE_ENGLISH_SUICIDE:
        print("  ⏩ Пропущен (INCLUDE_ENGLISH_SUICIDE = False)")
        return pd.DataFrame(columns=['text', 'label'])
    folder = os.path.join(DATASETS_DIR, "suicide-watch")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'text', 'comment', 'post'})
        lc = find_col(df, {'class', 'label', 'category'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(
                lambda x: 2 if str(x).lower().strip() == 'suicide' else 4)
            df = filter_df(df)
            s = df[df['label'] == 2].sample(n=min(3000, len(df[df['label'] == 2])), random_state=RANDOM_SEED)
            n = df[df['label'] == 4].sample(n=min(1000, len(df[df['label'] == 4])), random_state=RANDOM_SEED)
            df = pd.concat([s, n]).reset_index(drop=True)
            print(f"  ✅ {os.path.basename(f)} — ⚠️ АНГЛИЙСКИЙ!")
            print_stats(df, "Suicide Watch (EN)")
            return df
    return pd.DataFrame(columns=['text', 'label'])


def load_jigsaw():
    header(6, "Jigsaw Multilingual (русская часть)",
           "kaggle.com/c/jigsaw-multilingual-toxic-comment-classification")
    folder = os.path.join(DATASETS_DIR, "jigsaw-multilingual")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        if 'lang' in df.columns:
            df = df[df['lang'] == 'ru']
        tc = find_col(df, {'comment_text', 'text', 'comment'})
        lc = find_col(df, {'toxic', 'label', 'is_toxic'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(lambda x: 1 if float(x) >= 0.5 else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            print_stats(df, "Jigsaw (RU)")
            return df
    return pd.DataFrame(columns=['text', 'label'])


def load_gazeta():
    header(7, "Gazeta.ru (нейтральный контент)",
           "huggingface.co/datasets/IlyaGusev/gazeta")
    folder = os.path.join(DATASETS_DIR, "gazeta")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден (опционально) → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    dfs = []
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'text', 'summary', 'title', 'content', 'article'})
        if tc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = 4
            df = filter_df(df)
            df = df.sample(n=min(5000, len(df)), random_state=RANDOM_SEED)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['text', 'label'])
    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "Gazeta (neutral)")
    return df


def load_fakespeak():
    header(8, "Fakespeak-RUS (дезинформация)",
           "По запросу: researchgate.net/publication/394090726")
    folder = os.path.join(DATASETS_DIR, "fakespeak-rus")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден (по запросу) → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    dfs = []
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'text', 'content', 'article', 'news'})
        lc = find_col(df, {'label', 'class', 'fake', 'is_fake', 'veracity'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(
                lambda x: 3 if str(x).lower().strip() in ('fake', '1', 'true', 'false_news', 'disinformation') else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['text', 'label'])
    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "Fakespeak-RUS")
    return df


def load_vk_suicidal():
    header(9, "VK Suicidal Posts (суицид)",
           "По запросу: doi.org/10.1007/978-3-030-63119-2_66")
    folder = os.path.join(DATASETS_DIR, "vk-suicidal-posts")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден (по запросу) → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    dfs = []
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'text', 'post', 'message', 'content'})
        lc = find_col(df, {'label', 'class', 'suicidal', 'is_suicidal', 'category'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(
                lambda x: 2 if str(x).lower().strip() in ('suicide', 'suicidal', '1', 'true', 'yes') else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['text', 'label'])
    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "VK Suicidal")
    return df


def load_ethnicity_hate():
    header(10, "Ethnicity Hate Speech (ненависть)",
           "По запросу: doi.org/10.1016/j.ipm.2021.102615")
    folder = os.path.join(DATASETS_DIR, "ethnicity-hate-speech")
    files = find_csv(folder)
    if not files:
        print(f"  ⚠️  Не найден (по запросу) → {folder}/")
        return pd.DataFrame(columns=['text', 'label'])
    dfs = []
    for f in files:
        df = try_read_csv(f)
        if df is None:
            continue
        tc = find_col(df, {'text', 'message', 'comment', 'post'})
        lc = find_col(df, {'label', 'class', 'attitude', 'sentiment', 'hate'})
        if tc and lc:
            df = df.rename(columns={tc: 'text'})
            df['label'] = df[lc].apply(
                lambda x: 1 if str(x).lower().strip() in ('negative', 'hate', 'hateful', '1', '2') else 4)
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['text', 'label'])
    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "Ethnicity Hate")
    return df


def load_extras():
    header("★", "Дополнительные файлы в datasets/")
    known = {'manual', 'russian-language-toxic-comments',
             'toxic-russian-comments-pikabu-2ch', 'inappropriate-messages',
             'suicide-watch', 'jigsaw-multilingual', 'gazeta',
             'fakespeak-rus', 'vk-suicidal-posts', 'ethnicity-hate-speech'}

    root_csvs = [f for f in glob.glob(os.path.join(DATASETS_DIR, "*.csv"))
                 if os.path.basename(f) not in (FINAL_DATASET, TRAINING_DATASET)]

    extra_dirs = []
    if os.path.isdir(DATASETS_DIR):
        for item in os.listdir(DATASETS_DIR):
            full = os.path.join(DATASETS_DIR, item)
            if os.path.isdir(full) and item.lower() not in known:
                csvs = find_csv(full)
                if csvs:
                    extra_dirs.append((item, csvs))

    if not root_csvs and not extra_dirs:
        print("  Дополнительных файлов нет")
        return pd.DataFrame(columns=['text', 'label'])

    dfs = []
    for f in root_csvs:
        df = try_read_csv(f)
        if df is not None and 'text' in df.columns and 'label' in df.columns:
            df = filter_df(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,})")
            dfs.append(df)

    for folder_name, csvs in extra_dirs:
        for f in csvs:
            df = try_read_csv(f)
            if df is not None and 'text' in df.columns and 'label' in df.columns:
                df = filter_df(df)
                print(f"  ✅ {folder_name}/{os.path.basename(f)} ({len(df):,})")
                dfs.append(df)

    if not dfs:
        print("  Подходящих файлов нет (нужны колонки text, label)")
        return pd.DataFrame(columns=['text', 'label'])

    df = pd.concat(dfs, ignore_index=True)
    df = filter_df(df)
    print_stats(df, "Доп. данные")
    return df


# ============================================================================
# ОБЪЕДИНЕНИЕ
# ============================================================================

def combine_all():
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║  ПОДГОТОВКА ДАТАСЕТОВ ДЛЯ ОБУЧЕНИЯ RuBERT                    ║
║  Классы: насилие, ненависть, суицид, дезинформация, нейтр.    ║
╠════════════════════════════════════════════════════════════════╣
║  Папка: {DATASETS_DIR[:55]:55s}║
╚════════════════════════════════════════════════════════════════╝
    """)

    if not os.path.isdir(DATASETS_DIR):
        print(f"❌ Папка не найдена: {DATASETS_DIR}")
        sys.exit(1)

    print(f"📂 Содержимое:")
    for item in sorted(os.listdir(DATASETS_DIR)):
        full = os.path.join(DATASETS_DIR, item)
        if os.path.isdir(full):
            print(f"   📁 {item}/ ({len(os.listdir(full))} файлов)")
        elif item.endswith('.csv'):
            print(f"   📄 {item} ({os.path.getsize(full)/1024:.0f} KB)")

    # --- Загрузка ---
    loaders = [
        ("manual",         load_manual_dataset),
        ("kaggle_toxic",   load_kaggle_toxic),
        ("pikabu_2ch",     load_pikabu_2ch_ext),
        ("inappropriate",  load_inappropriate),
        ("suicide_reddit", load_suicide_watch),
        ("jigsaw",         load_jigsaw),
        ("gazeta",         load_gazeta),
        ("fakespeak",      load_fakespeak),
        ("vk_suicide",     load_vk_suicidal),
        ("ethnicity_hate", load_ethnicity_hate),
        ("extra",          load_extras),
    ]

    all_dfs = []
    stats = {}

    for name, fn in loaders:
        df = fn()
        if len(df) > 0:
            df['source'] = name
            all_dfs.append(df)
            stats[name] = len(df)

    if not all_dfs:
        print("\n❌ Ни один датасет не загружен!")
        print("   Скачайте датасеты — см. DATASETS_README.md")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)
    combined = shuffle(combined, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n{'═'*60}")
    print(f"  🔄 ОБЪЕДИНЕНИЕ")
    print(f"{'═'*60}")
    print(f"  Источников: {len(stats)}")
    for s, c in stats.items():
        print(f"    • {s}: {c:,}")
    print(f"  Дубликатов удалено: {before - len(combined):,}")
    print_stats(combined, "ИТОГО (до балансировки)")

    # --- Балансировка ---
    if TARGET_PER_CLASS:
        print(f"\n  ⚖️  Балансировка: {TARGET_PER_CLASS}/класс")
        parts = []
        for label in sorted(combined['label'].unique()):
            cls = combined[combined['label'] == label]
            if len(cls) >= TARGET_PER_CLASS:
                parts.append(cls.sample(n=TARGET_PER_CLASS, random_state=RANDOM_SEED))
            else:
                parts.append(cls)
                print(f"    ⚠️  {LABEL_NAMES[label]}: {len(cls)} < {TARGET_PER_CLASS}")
        combined = shuffle(pd.concat(parts, ignore_index=True), random_state=RANDOM_SEED).reset_index(drop=True)
        print_stats(combined, "ИТОГО (после балансировки)")

    # --- Сохранение ---
    full_path = os.path.join(OUTPUT_DIR, FINAL_DATASET)
    combined.to_csv(full_path, index=False, encoding='utf-8')

    clean_path = os.path.join(OUTPUT_DIR, TRAINING_DATASET)
    combined[['text', 'label']].to_csv(clean_path, index=False, encoding='utf-8')

    print(f"""
╔════════════════════════════════════════════════════════════════╗
║  ✅ ГОТОВО!                                                    ║
╠════════════════════════════════════════════════════════════════╣
║  Примеров:   {len(combined):>7,}                                       ║
║  Классов:    {combined['label'].nunique():>7}                                       ║
║  Источников: {len(stats):>7}                                       ║
╠════════════════════════════════════════════════════════════════╣
║  Файлы:                                                       ║
║  • {FINAL_DATASET:55s}  ║
║  • {TRAINING_DATASET:55s}  ║
╠════════════════════════════════════════════════════════════════╣
║  В НОУТБУКЕ:                                                  ║
║  df = pd.read_csv(r'{clean_path}')
╚════════════════════════════════════════════════════════════════╝
    """)
    return combined


if __name__ == "__main__":
    combine_all()

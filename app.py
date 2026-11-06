"""Демо-приложение: отбор хороших вин по лабораторному анализу.

Модель обучена в ноутбуке HW3_MMDA.ipynb, здесь она только загружается.

Почему разные декораторы кэша:
- st.cache_resource хранит один общий объект (модель) без копирования.
  Модель не меняется, её не нужно сериализовать и копировать для каждого
  пользователя, а повторное чтение с диска при каждом перезапуске скрипта
  было бы лишней работой.
- st.cache_data хранит данные (таблицы, массивы вероятностей) и отдаёт
  каждому вызову копию результата. Если код случайно изменит таблицу,
  кэш не испортится.
- Без декоратора joblib.load выполнялся бы при каждом действии пользователя
  (сдвиг слайдера, ввод числа): модель заново читалась бы с диска,
  и приложение работало бы медленнее.
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / 'model.joblib'
CONFIG_PATH = APP_DIR / 'config.json'
TEST_PATH = APP_DIR / 'test.csv'

# цвета графиков
BLUE, ORANGE, GRAY, INK = '#2a78d6', '#eb6834', '#898781', '#52514e'

FEATURE_INFO = {
    'fixed_acidity': ('Фиксированная кислотность', 'г/дм³'),
    'volatile_acidity': ('Летучая кислотность', 'г/дм³'),
    'citric_acid': ('Лимонная кислота', 'г/дм³'),
    'residual_sugar': ('Остаточный сахар', 'г/дм³'),
    'chlorides': ('Хлориды', 'г/дм³'),
    'free_sulfur_dioxide': ('Свободный SO₂', 'мг/дм³'),
    'total_sulfur_dioxide': ('Общий SO₂', 'мг/дм³'),
    'density': ('Плотность', 'г/см³'),
    'pH': ('pH', ''),
    'sulphates': ('Сульфаты', 'г/дм³'),
    'alcohol': ('Алкоголь', '% об.'),
    'is_red': ('Цвет вина', ''),
}


# ---------- загрузка и кэш ----------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text())


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):
    # считается один раз, от порога не зависит
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def test_curves(y, proba):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    prec, rec, _ = metrics.precision_recall_curve(y, proba)
    return {
        'fpr': fpr, 'tpr': tpr, 'precision': prec, 'recall': rec,
        'roc_auc': metrics.roc_auc_score(y, proba),
        'pr_auc': metrics.average_precision_score(y, proba),
    }


@st.cache_data
def batch_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def feature_label(name):
    label, unit = FEATURE_INFO.get(name, (name, ''))
    return f'{label}, {unit}' if unit else label


def confusion_at(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    y = np.asarray(y)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision,
            'recall': recall, 'f1': f1, 'fpr': fpr, 'selected': tp + fp}


# ---------- страница ----------

st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')

model = load_model()
config = load_config()
test_df = load_test()
FEATURES = list(model.feature_names_in_)
X_test, y_test = test_df[FEATURES], test_df['target']
proba_test = test_proba(model, X_test)
default_threshold = float(config['threshold'])

if 'threshold' not in st.session_state:
    st.session_state.threshold = default_threshold


def reset_threshold():
    st.session_state.threshold = default_threshold


with st.sidebar:
    st.header('Порог решения')
    st.slider('Вино попадает в подборку, если вероятность не ниже порога',
              min_value=0.0, max_value=1.0, step=0.005, format='%.3f', key='threshold')
    st.button(f'Вернуть порог из config.json ({default_threshold:.3f})', on_click=reset_threshold)
    st.caption('Порог общий для всех вкладок.')
    st.divider()
    st.markdown(
        f'**Модель:** логистическая регрессия со стандартизацией признаков.\n\n'
        f'**Тестовая выборка:** {len(test_df)} вин, из них хороших '
        f'{int(y_test.sum())} ({y_test.mean():.1%}).'
    )

threshold = st.session_state.threshold

st.title('🍷 Отбор хороших вин для подписки')
st.caption('Модель оценивает вероятность того, что эксперты поставят вину оценку 7 или выше, '
           'по результатам лабораторного анализа.')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])


# ---------- 4.1 одно вино ----------

with tab_one:
    st.subheader('Характеристики вина')
    st.caption('Диапазоны полей взяты из тестовой выборки, значения по умолчанию равны медианам.')

    values = {}
    numeric = [f for f in FEATURES if f != 'is_red']
    cols = st.columns(3)
    for i, name in enumerate(numeric):
        col_data = test_df[name]
        lo, hi, med = float(col_data.min()), float(col_data.max()), float(col_data.median())
        step = 10 ** np.floor(np.log10((hi - lo) / 100)) if hi > lo else 0.01
        decimals = max(0, int(-np.log10(step)))
        with cols[i % 3]:
            values[name] = st.number_input(
                feature_label(name), min_value=lo, max_value=hi, value=med,
                step=float(step), format=f'%.{decimals}f', key=f'in_{name}')
    with cols[len(numeric) % 3]:
        color = st.radio('Цвет вина', ['белое', 'красное'], horizontal=True,
                         index=int(test_df['is_red'].median()))
    values['is_red'] = int(color == 'красное')

    wine = pd.DataFrame([values])[FEATURES]
    p_good = float(model.predict_proba(wine)[0, 1])

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric('Вероятность, что вино хорошее', f'{p_good:.1%}',
                  delta=f'{p_good - threshold:+.1%} к порогу', delta_color='normal')
    with c2:
        if p_good >= threshold:
            st.success(f'Решение модели: **в подборку**. Вероятность {p_good:.3f} не ниже порога {threshold:.3f}.')
        else:
            st.warning(f'Решение модели: **не в подборку**. Вероятность {p_good:.3f} ниже порога {threshold:.3f}.')
        st.progress(min(max(p_good, 0.0), 1.0))

    with st.expander('Какие признаки повлияли на оценку этого вина'):
        scaler, clf = model[0], model[-1]
        contrib = pd.Series(scaler.transform(wine)[0] * clf.coef_[0], index=FEATURES)
        contrib = contrib.rename(index=feature_label).sort_values()
        st.caption('Вклад признака в логит: коэффициент модели, умноженный на стандартизованное значение. '
                   'Положительный вклад повышает вероятность, отрицательный понижает.')
        st.dataframe(contrib.to_frame('вклад в логит').round(3), width='stretch')


# ---------- 4.2 метрики ----------

with tab_metrics:
    cur = confusion_at(y_test, proba_test, threshold)
    ref = confusion_at(y_test, proba_test, default_threshold)
    curves = test_curves(y_test, proba_test)

    st.subheader(f'Качество на тестовой выборке при пороге {threshold:.3f}')
    st.caption(f'Изменения показаны относительно порога из config.json ({default_threshold:.3f}).')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Precision', f"{cur['precision']:.3f}", delta=f"{cur['precision'] - ref['precision']:+.3f}")
    m2.metric('Recall', f"{cur['recall']:.3f}", delta=f"{cur['recall'] - ref['recall']:+.3f}")
    m3.metric('F1', f"{cur['f1']:.3f}", delta=f"{cur['f1'] - ref['f1']:+.3f}")
    m4.metric('Вин в подборке', cur['selected'], delta=cur['selected'] - ref['selected'],
              delta_color='off')

    c_cm, c_roc, c_pr = st.columns(3)
    with c_cm:
        cm = np.array([[cur['tn'], cur['fp']], [cur['fn'], cur['tp']]])
        fig, ax = plt.subplots(figsize=(4, 3.6))
        metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее']).plot(
            ax=ax, cmap='Blues', colorbar=False)
        ax.set_xlabel('решение модели')
        ax.set_ylabel('оценка экспертов')
        ax.set_title('Матрица ошибок')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c_roc:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(curves['fpr'], curves['tpr'], color=BLUE, lw=2, label=f"ROC-AUC = {curves['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], color=GRAY, ls='--', lw=1)
        ax.scatter(cur['fpr'], cur['recall'], s=80, color=ORANGE, edgecolor='white', lw=2, zorder=3,
                   label=f'порог {threshold:.3f}')
        ax.set(xlabel='FPR', ylabel='TPR (recall)', title='ROC-кривая')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c_pr:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(curves['recall'], curves['precision'], color=BLUE, lw=2,
                label=f"PR-AUC = {curves['pr_auc']:.3f}")
        ax.axhline(y_test.mean(), color=GRAY, ls='--', lw=1, label='случайная модель')
        ax.scatter(cur['recall'], cur['precision'], s=80, color=ORANGE, edgecolor='white', lw=2,
                   zorder=3, label=f'порог {threshold:.3f}')
        ax.set(xlabel='recall', ylabel='precision', title='PR-кривая', ylim=(0, 1.02))
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    good_total = int(y_test.sum())
    st.info(
        f"**Что это значит для подборки.** Сейчас в подборку попадает {cur['selected']} вин. "
        f"Из них {cur['tp']} действительно хорошие, а {cur['fp']} посредственные. "
        f"Ещё {cur['fn']} хороших вин из {good_total} в подборку не попали.\n\n"
        "- Если **увеличить порог**, подборка станет короче и надёжнее: посредственных вин будет меньше "
        "(precision растёт), но часть хороших вин тоже выпадет (recall падает).\n"
        "- Если **уменьшить порог**, подборка станет шире: найдётся больше хороших вин (recall растёт), "
        "но вместе с ними попадёт больше посредственных (precision падает)."
    )


# ---------- 4.3 пакетное предсказание ----------

with tab_batch:
    st.subheader('Оценка списка вин из CSV-файла')
    st.markdown(f'Файл должен содержать столбцы: `{"`, `".join(FEATURES)}`. '
                f'Столбец `target` не нужен. Используется порог из сайдбара: **{threshold:.3f}**.')
    example = test_df[FEATURES].head(20).to_csv(index=False).encode('utf-8')
    st.download_button('Скачать пример файла (20 вин)', example, file_name='wines_example.csv',
                       mime='text/csv')

    uploaded = st.file_uploader('CSV-файл с признаками вин', type='csv')
    if uploaded is not None:
        try:
            batch = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as exc:
            batch = None
            st.error(f'Не удалось прочитать файл как CSV: {exc}')

        if batch is not None:
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error(f'В файле не хватает столбцов: {", ".join(missing)}. '
                         'Добавьте их и загрузите файл снова.')
            elif batch.empty:
                st.error('Файл пустой: в нём нет ни одной строки с вином.')
            else:
                X_batch = batch[FEATURES].apply(pd.to_numeric, errors='coerce')
                bad_rows = X_batch.isna().any(axis=1)
                if bad_rows.all():
                    st.error('Ни в одной строке нет полного набора числовых признаков.')
                else:
                    if bad_rows.any():
                        st.warning(f'Строк с пропусками или нечисловыми значениями: {int(bad_rows.sum())}. '
                                   'Для них прогноз не считается.')
                    proba = pd.Series(np.nan, index=batch.index)
                    proba[~bad_rows] = batch_proba(model, X_batch[~bad_rows])
                    result = batch.copy()
                    result['proba_good'] = proba.round(4)
                    result['decision'] = np.where(proba.isna(), 'нет данных',
                                                  np.where(proba >= threshold, 'в подборку', 'не в подборку'))
                    n_sel = int((result['decision'] == 'в подборку').sum())
                    st.success(f'Оценено вин: {int((~bad_rows).sum())}. В подборку попадает: {n_sel}.')
                    st.dataframe(result, width='stretch')
                    st.download_button('Скачать результат (CSV)',
                                       result.to_csv(index=False).encode('utf-8'),
                                       file_name='wine_predictions.csv', mime='text/csv')


# ---------- 4.4 коэффициенты ----------

with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    st.caption('Признаки стандартизованы, поэтому коэффициенты можно сравнивать между собой: '
               'это изменение логита при росте признака на одно стандартное отклонение.')
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [feature_label(f) for f in coefs.index]
    ax.barh(labels, coefs.values, color=[BLUE if v > 0 else ORANGE for v in coefs.values])
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel('коэффициент (после стандартизации)')
    ax.set_title('Синие признаки голосуют за хорошее вино, оранжевые против')
    ax.grid(axis='x', color='#e1e0d9', lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    pos = coefs[coefs > 0].sort_values(ascending=False).head(3)
    neg = coefs[coefs < 0].head(3)
    c1, c2 = st.columns(2)
    c1.markdown('**Сильнее всего за хорошее вино:**\n' +
                '\n'.join(f'- {feature_label(f)}: {v:+.2f}' for f, v in pos.items()))
    c2.markdown('**Сильнее всего против:**\n' +
                '\n'.join(f'- {feature_label(f)}: {v:+.2f}' for f, v in neg.items()))
    st.caption('Коэффициенты показывают связь при прочих равных признаках. '
               'Сильно связанные признаки (например, плотность, сахар и алкоголь) делят влияние между собой, '
               'поэтому знак отдельного коэффициента не стоит читать как причину.')

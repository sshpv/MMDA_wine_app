import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
PINK = '#B4466A'


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    # считается один раз
    return _model.predict_proba(X)[:, 1]


def meme(name, width=220):
    path = APP_DIR / 'images' / f'{name}.jpg'
    if path.exists():
        st.image(str(path), width=width)


st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')

model = load_model()
default_threshold = json.loads((APP_DIR / 'config.json').read_text())['threshold']
test_df = load_test()
FEATURES = list(model.feature_names_in_)
X_test, y_test = test_df[FEATURES], test_df['target'].to_numpy()
proba_test = test_proba(model, X_test)

with st.sidebar:
    threshold = st.slider('Порог', 0.0, 1.0, float(default_threshold), step=0.001, format='%.3f',
                          key='threshold', help='Вино идёт в подборку, если P(хорошее) ≥ порога')
    st.caption(f'По умолчанию {default_threshold:.3f}: порог, максимизирующий F1 на валидации')

st.title('Подборка вин')
st.markdown('### *от логистической регрессии* 🍷')

tab1, tab2, tab3, tab4 = st.tabs(['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

# 4.1 
with tab1:
    stats = X_test.describe()
    values = {'is_red': int(st.radio('Тип', ['белое', 'красное'], horizontal=True) == 'красное')}
    cols = st.columns(3)
    for i, f in enumerate(f for f in FEATURES if f != 'is_red'):
        lo, hi, med = stats.loc['min', f], stats.loc['max', f], stats.loc['50%', f]
        fmt, step = ('%.5f', 0.0001) if f == 'density' else ('%.3f', None)
        values[f] = cols[i % 3].number_input(f, float(lo), float(hi), float(med), step=step, format=fmt)

    p = model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1]
    with st.container(border=True):
        left, right = st.columns([3, 1])
        left.metric('Вероятность, что вино хорошее', f'{p:.1%}')
        if p >= threshold:
            left.success(f'Берём в подборку ({p:.1%} ≥ {threshold:.3f})')
            with right:
                meme('good')
        else:
            left.error(f'Не берём ({p:.1%} < {threshold:.3f})')
            with right:
                meme('bad')

# 4.2
with tab2:
    pred = (proba_test >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    prec = metrics.precision_score(y_test, pred, zero_division=0)
    rec = metrics.recall_score(y_test, pred)

    c = st.columns(4)
    c[0].metric('Precision', f'{prec:.3f}')
    c[1].metric('Recall', f'{rec:.3f}')
    c[2].metric('F1', f'{metrics.f1_score(y_test, pred, zero_division=0):.3f}')
    c[3].metric('Вин в подборке', f'{tp + fp} из {len(y_test)}')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics.ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]), display_labels=['не хорошее', 'хорошее']).plot(
        ax=ax[0], cmap='PuRd', colorbar=False, values_format='d')
    fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
    ax[1].plot(fpr, tpr, color=PINK, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, proba_test):.3f}')
    ax[1].plot([0, 1], [0, 1], 'k:')
    ax[1].scatter(fp / (fp + tn), rec, s=80, c='k', zorder=3, label='текущий порог')
    pr, rc, _ = metrics.precision_recall_curve(y_test, proba_test)
    ax[2].plot(rc, pr, color=PINK, label=f'PR-AUC = {metrics.average_precision_score(y_test, proba_test):.3f}')
    ax[2].axhline(y_test.mean(), color='k', ls=':')
    if tp + fp > 0:
        ax[2].scatter(rec, prec, s=80, c='k', zorder=3, label='текущий порог')
    ax[0].set_title('Матрица ошибок')
    ax[1].set(title='ROC', xlabel='FPR', ylabel='TPR')
    ax[2].set(title='PR', xlabel='Recall', ylabel='Precision')
    ax[1].legend(loc='lower right')
    ax[2].legend()
    st.pyplot(fig)

    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.markdown(
            '#### *Памятка для сомелье*\n'
            f'Сейчас в подборке {tp + fp} вин, хороших среди них {prec:.0%}, '
            f'и это {rec:.0%} всех хороших вин.\n\n'
            '- **Порог выше**: подборка меньше, посредственных вин в ней реже (precision ↑), '
            'но больше хороших остаётся за бортом (recall ↓).\n'
            '- **Порог ниже**: подборка шире, хороших в неё попадает больше (recall ↑), '
            'но и посредственных тоже (precision ↓).')
        with right:
            meme('sommelier', width=170)

# 4.3 
with tab3:
    st.write(f'CSV со столбцами: `{", ".join(FEATURES)}`. Порог из сайдбара: **{threshold:.3f}**')
    file = st.file_uploader('Загрузите CSV', type='csv')
    if file is None:
        meme('waiting', width=180)
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            df = None
            st.error(f'Не удалось прочитать файл: {e}')
        missing = [] if df is None else [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f'Не хватает столбцов: {", ".join(missing)}')
        elif df is not None:
            X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
            ok = X.notna().all(axis=1)
            if not ok.any():
                st.error('Нет ни одной строки с корректными числовыми признаками')
            else:
                if not ok.all():
                    st.warning(f'Строк с пропусками или нечисловыми значениями: {(~ok).sum()}. Они пропущены')
                df.insert(0, 'proba_good', np.nan)
                df.loc[ok, 'proba_good'] = model.predict_proba(X[ok])[:, 1].round(4)
                df.insert(1, 'recommend', np.where(df['proba_good'] >= threshold, 'да', 'нет'))
                df.loc[~ok, 'recommend'] = None
                st.dataframe(df)
                st.download_button('Скачать CSV', df.to_csv(index=False).encode('utf-8'), 'wine_predictions.csv')

# 4.4 
with tab4:
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(coefs.index, coefs.values, color=np.where(coefs > 0, PINK, '#C9C3C6'))
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('коэффициент (признаки стандартизованы)')
    st.pyplot(fig)
    st.markdown(f'**За хорошее вино** сильнее всего: {", ".join(coefs.index[::-1][:3])}.  \n'
                f'**Против**: {", ".join(coefs.index[:3])}.')
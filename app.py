import os
import json
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_ENABLED = True
except ImportError:
    GENAI_ENABLED = False

try:
    from scapy.all import rdpcap, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

st.set_page_config(page_title="🚀 Deep Cybersecurity ML Analyzer with Gemini", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Ошибка загрузки CSV: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def process_pcap(file):
    if not SCAPY_AVAILABLE:
        st.error("Scapy не установлен. Установите через `pip install scapy`.")
        return pd.DataFrame()
    content = file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        packets = rdpcap(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        st.error(f"Ошибка чтения PCAP: {e}")
        return pd.DataFrame()
    os.unlink(tmp_path)

    records = []
    for i, pkt in enumerate(packets):
        if IP in pkt:
            rec = {
                "packet_id": i,
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "protocol": pkt[IP].proto,
                "length": len(pkt),
                "ttl": pkt[IP].ttl,
                "timestamp": pkt.time if hasattr(pkt, 'time') else np.nan,
            }
            if TCP in pkt:
                rec.update({
                    "src_port": pkt[TCP].sport,
                    "dst_port": pkt[TCP].dport,
                    "tcp_flags": int(pkt[TCP].flags),
                    "window_size": pkt[TCP].window,
                })
            elif UDP in pkt:
                rec.update({
                    "src_port": pkt[UDP].sport,
                    "dst_port": pkt[UDP].dport,
                    "udp_length": pkt[UDP].len,
                })
            records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        df["flow_id"] = df.apply(
            lambda x: f"{x['src_ip']}:{x.get('src_port',0)}->{x['dst_ip']}:{x.get('dst_port',0)}", axis=1)
        # Добавим временные признаки в рамках потока:
        df = df.groupby("flow_id").agg({
            "length": ["mean", "sum", "count"],
            "ttl": ["mean", "min", "max"],
            "timestamp": ["min", "max", "std"],
            "tcp_flags": "mean",
            "window_size": "mean",
            "udp_length": "mean"
        }).reset_index()
        df.columns = ["_".join(col).strip("_") for col in df.columns.values]
        df["flow_duration"] = df["timestamp_max"] - df["timestamp_min"]
        df["flow_timestamp_std"] = df["timestamp_std"].fillna(0)
    return df

def preprocess_features(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.columns.tolist()

def generate_analysis_summary(atype, params, metrics):
    lines = [f"Тип анализа: {atype}", f"Параметры: {params}", "Метрики:"]
    for k,v in metrics.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

def call_gemini_with_retries(prompt, max_retries=3):
    if not GENAI_ENABLED:
        st.warning("Gemini API недоступен")
        return {}
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    for i in range(max_retries):
        try:
            response = model.generate_content(prompt, timeout=15)
            j = json.loads(response.text)
            if j:
                return j
        except Exception as e:
            st.warning(f"Ошибка Gemini, попытка {i+1}: {e}")
    st.error("Не удалось получить ответ от Gemini после повторных попыток")
    return {}

def plot_interactive_pca(X_scaled, labels, title):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df_plot = pd.DataFrame({'PC1': pcs[:,0], 'PC2': pcs[:,1], 'Label': labels.astype(str)})
    fig = px.scatter(df_plot, x='PC1', y='PC2', color='Label', title=title, labels={'PC1':'PC1','PC2':'PC2'}, hover_data=df_plot.columns)
    st.plotly_chart(fig, use_container_width=True)

def explain_classification_model(clf, X_train, feature_names):
    if not SHAP_AVAILABLE:
        st.warning("Установите SHAP для объяснения модели: pip install shap")
        return
    st.subheader("🧩 Объяснение классификации с SHAP")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False)
    st.pyplot(bbox_inches='tight')

def analyze_tcp_flags(df):
    st.subheader("🔍 Анализ TCP-флагов")
    if "tcp_flags_mean" not in df.columns:
        st.write("Нет данных TCP флагов")
        return
    df["syn_flag"] = df["tcp_flags_mean"].apply(lambda x: (int(x) & 0x02) > 0)
    syn_count = df["syn_flag"].sum()
    st.write(f"Потоков с SYN-флагом: {syn_count} из {len(df)}")
    st.dataframe(df[["flow_id", "tcp_flags_mean", "syn_flag"]].head(10))

def behavioral_analysis(df):
    st.subheader("📈 Поведенческий анализ потоков")
    if "flow_duration" in df.columns:
        fig = px.histogram(df, x="flow_duration", nbins=30, title="Длительность потоков (секунды)")
        st.plotly_chart(fig, use_container_width=True)
    if "length_sum" in df.columns and "length_count" in df.columns:
        df["avg_packet_size"] = df["length_sum"] / df["length_count"]
        fig2 = px.histogram(df, x="avg_packet_size", nbins=30, title="Средний размер пакета в потоке")
        st.plotly_chart(fig2, use_container_width=True)

def kb_report(df):
    st.subheader("🛡️ Итоговый отчет по безопасности")
    if "flow_duration" in df.columns and "length_count" in df.columns:
        suspect = df[(df["flow_duration"] < 0.1) & (df["length_count"] > df["length_count"].median())]
        st.write(f"Обнаружено подозрительных потоков: {len(suspect)}")
        st.dataframe(suspect.head(10))
    else:
        st.warning("Отсутствуют необходимые колонки 'flow_duration' и/или 'length_count' для анализа подозрительных потоков.")
        st.write("Доступные колонки:", df.columns.tolist())

# --- Основной UI ---

st.title("🚀 Глубокий КБ и ML анализ сетевого трафика с Gemini")

file_type = st.sidebar.selectbox("Тип файла", ["CSV", "PCAP"])
uploaded_file = st.sidebar.file_uploader("Загрузите файл", type=["csv"] if file_type=="CSV" else ["pcap", "pcapng"])

if uploaded_file:
    df = load_csv(uploaded_file) if file_type=="CSV" else process_pcap(uploaded_file)
    if df.empty:
        st.error("Ошибка загрузки данных")
        st.stop()

    st.subheader("Пример данных")
    st.dataframe(df.head())

    id_col = st.sidebar.selectbox("ID колонка", options=df.columns, index=0)
    default_feats = df.select_dtypes(include=np.number).columns.tolist()
    if id_col in default_feats:
        default_feats.remove(id_col)
    feat_cols = st.sidebar.multiselect("Признаки для анализа", options=[c for c in df.columns if c!=id_col], default=default_feats[:5])

    if len(feat_cols) == 0:
        st.warning("Выберите хотя бы один признак")
        st.stop()

    analysis_type = st.sidebar.radio("Тип анализа", ["Clustering", "Classification", "Anomaly Detection"])

    if "ml_params" not in st.session_state:
        st.session_state.ml_params = {"n_clusters": 3, "contamination": 0.1, "n_estimators": 100}

    st.sidebar.header("Параметры модели")
    if analysis_type == "Clustering":
        st.session_state.ml_params["n_clusters"] = st.sidebar.slider("Число кластеров", 2, 15, st.session_state.ml_params["n_clusters"])
    elif analysis_type == "Anomaly Detection":
        st.session_state.ml_params["contamination"] = st.sidebar.slider("Доля аномалий", 0.01, 0.5, st.session_state.ml_params["contamination"], 0.01)
    elif analysis_type == "Classification":
        st.session_state.ml_params["n_estimators"] = st.sidebar.slider("Число деревьев", 10, 300, st.session_state.ml_params["n_estimators"], 10)

    # КБ анализ
    analyze_tcp_flags(df)
    behavioral_analysis(df)
    kb_report(df)

    X_scaled, feature_names = preprocess_features(df, feat_cols)

    if st.button("Запустить ML-анализ"):
        if analysis_type == "Clustering":
            model = KMeans(n_clusters=st.session_state.ml_params["n_clusters"], random_state=42)
            labels = model.fit_predict(X_scaled)
            df['Cluster'] = labels
            sil_score = silhouette_score(X_scaled, labels)
            st.write(f"Silhouette score: {sil_score:.4f}")
            st.bar_chart(df['Cluster'].value_counts())
            plot_interactive_pca(X_scaled, labels, f"KMeans Clustering (k={st.session_state.ml_params['n_clusters']}) PCA")

            summary = generate_analysis_summary(analysis_type, st.session_state.ml_params, {"silhouette_score": sil_score})

        elif analysis_type == "Classification":
            target_cols = [c for c in df.columns if c not in feat_cols + [id_col]]
            if not target_cols:
                st.error("Нет целевого столбца для классификации")
                st.stop()
            target_col = st.selectbox("Целевой столбец", target_cols)
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=st.session_state.ml_params["n_estimators"], random_state=42)
            clf.fit(X_train, y_train)

            # Кросс-валидация и метрики
            scores = cross_val_score(clf, X_scaled, y, cv=5)
            st.write(f"Cross-Validation Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.subheader("Отчет классификации")
            st.dataframe(pd.DataFrame(report).transpose())
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Важность признаков")
            st.plotly_chart(fig, use_container_width=True)

            pred_labels = clf.predict(X_scaled)
            plot_interactive_pca(X_scaled, pred_labels, "Random Forest Classification PCA")

            # SHAP объяснение
            explain_classification_model(clf, X_train, feature_names)

            summary = generate_analysis_summary(analysis_type, st.session_state.ml_params, {"accuracy": report.get("accuracy", "N/A")})

        else:  # Anomaly Detection
            model = IsolationForest(contamination=st.session_state.ml_params["contamination"], random_state=42)
            preds = model.fit_predict(X_scaled)
            df['Anomaly'] = (preds == -1).astype(int)
            count_anom = df['Anomaly'].sum()
            total = len(df)
            anomaly_ratio = count_anom / total
            st.write(f"Обнаружено аномалий: {count_anom} из {total} ({anomaly_ratio:.2%})")
            st.dataframe(df[df['Anomaly'] == 1].head(20))
            plot_interactive_pca(X_scaled, df['Anomaly'], "Isolation Forest Anomaly Detection PCA")

            summary = generate_analysis_summary(analysis_type, st.session_state.ml_params, {
                "anomalies_detected": count_anom,
                "anomaly_ratio": f"{anomaly_ratio:.2%}"
            })

        st.subheader("Результаты ML-анализа")
        st.code(summary)

        if GENAI_ENABLED:
            st.subheader("Получаем рекомендации от Gemini...")
            new_params = call_gemini_with_retries(f"""
You are an expert ML engineer and cybersecurity analyst. Based on this analysis summary, recommend improved ML parameters as JSON:

{summary}

Return JSON with keys such as "n_clusters", "contamination", "n_estimators".
""")
            if new_params:
                st.success("Gemini рекомендует параметры:")
                st.json(new_params)

                n_clusters = int(new_params.get("n_clusters", st.session_state.ml_params["n_clusters"]))
                if n_clusters < 2 or n_clusters > 15:
                    n_clusters = st.session_state.ml_params["n_clusters"]
                contamination = float(new_params.get("contamination", st.session_state.ml_params["contamination"]))
                if contamination < 0.01 or contamination > 0.5:
                    contamination = st.session_state.ml_params["contamination"]
                n_estimators = int(new_params.get("n_estimators", st.session_state.ml_params["n_estimators"]))
                if n_estimators < 10 or n_estimators > 300:
                    n_estimators = st.session_state.ml_params["n_estimators"]

                st.session_state.ml_params.update({
                    "n_clusters": n_clusters,
                    "contamination": contamination,
                    "n_estimators": n_estimators
                })

                st.info("Параметры обновлены. Запустите анализ заново с новыми значениями.")
            else:
                st.warning("Не удалось получить рекомендации от Gemini.")
else:
    st.info("Загрузите CSV или PCAP файл для начала анализа.")

st.markdown("---")
st.caption("Powered by Streamlit, scikit-learn, SHAP & Google Gemini API")

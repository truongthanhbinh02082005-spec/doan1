import pandas as pd
import numpy as np
import warnings
import os 
import base64 
from io import BytesIO 
import json
from datetime import datetime  # <--- [MỚI] Thêm thư viện để lấy giờ

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

import shap

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid") 

# --- HÀM VẼ BIỂU ĐỒ PIE CHO SHAP ---
def create_pie_chart(factor_data, title):
    if factor_data.empty or factor_data.sum() == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, 'Không có đặc trưng\nđóng góp đáng kể',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='gray')
        ax.axis('off')
        return fig

    if len(factor_data) > 5:
        top_5 = factor_data.nlargest(5)
        other_sum = factor_data.nsmallest(len(factor_data) - 5).sum()
        if other_sum > 0:
            top_5['Các đặc trưng khác'] = other_sum
        factor_data = top_5

    labels = factor_data.index
    sizes = factor_data.values

    total = np.sum(sizes)
    legend_labels = [f'{label} ({(size/total)*100:.1f}%)' for label, size in zip(labels, sizes)]

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(sizes)))

    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct='%1.1f%%',      
        startangle=90,
        colors=colors,
        pctdistance=0.7,     
        labels=None,         
        labeldistance=None
    )
    
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.legend(
        wedges, 
        legend_labels, 
        title="Thuộc tính",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1), 
        ncol=min(len(labels), 3), 
        fontsize='small'
    )
    
    ax.axis('equal')
    plt.title(title, pad=20, fontsize=16)
    return fig

# --- HÀM KHỞI TẠO VÀ HUẤN LUYỆN MODEL ---
def train_model_on_startup():
    print("--- Bắt đầu quá trình huấn luyện mô hình và tạo biểu đồ ---")
    
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Đã tạo thư mục 'static' để chứa ảnh.")
    try:
        df = pd.read_csv('customer_churn_dataset-training-master.csv')
        print("Tải dữ liệu thành công.")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'customer_churn_dataset-training-master.csv'")
        exit()

    # 1. Xử lý dữ liệu thiếu
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("Làm sạch dữ liệu xong.")

    # 2. Tiền xử lý & Vẽ EDA
    customer_ids = df['CustomerID']
    df_features_raw = df.drop(columns=['CustomerID'])
    df_features_raw['Churn'] = df_features_raw['Churn'].astype(int)
    
    # Biểu đồ 1: Phân bổ Churn 
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df_features_raw, x='Churn', palette='pastel')
    plt.title('Phân bổ khách hàng Rời bỏ (1) vs Gắn bó (0)')
    plt.savefig('static/eda_01_churn_distribution.png')
    plt.close()

    # Biểu đồ 2 (Lặp): Numerical vs. Churn
    numerical_features_to_plot = ['Age', 'Tenure', 'Total Spend', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Last Interaction']
    for i, feature in enumerate(numerical_features_to_plot):
        plt.figure(figsize=(10, 6))
        if feature in ['Age', 'Tenure', 'Usage Frequency', 'Last Interaction']:
            sns.kdeplot(data=df_features_raw, x=feature, hue='Churn', fill=True, common_norm=False, palette='viridis')
            plt.title(f'Mật độ phân bổ {feature} theo Churn')
        else:
            sns.boxplot(data=df_features_raw, x='Churn', y=feature, palette='coolwarm')
            plt.title(f'So sánh {feature} giữa nhóm Churn và Gắn bó')
        plt.xlabel(feature)
        plt.savefig(f'static/eda_02_{i}_{feature.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Biểu đồ 3 : Categorical (Trước mã hóa)
    categorical_features_to_plot = ['Gender', 'Subscription Type', 'Contract Length']
    for i, feature in enumerate(categorical_features_to_plot):
        plt.figure(figsize=(10, 6))
        order = None
        if feature == 'Subscription Type': order = ['Basic', 'Standard', 'Premium']
        if feature == 'Contract Length': order = ['Monthly', 'Quarterly', 'Annual']
        sns.countplot(data=df_features_raw, x=feature, palette='ocean', order=order)
        plt.title(f'Phân bổ {feature} (Trước mã hóa)')
        plt.savefig(f'static/eda_03_{i}_{feature.lower().replace(" ", "_")}.png')
        plt.close()
    
    # 2B. MÃ HÓA DỮ LIỆU 
    df_features_encoded = pd.get_dummies(df_features_raw, columns=['Gender'], drop_first=True)
    global subscription_map, contract_map 
    subscription_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    contract_map = {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}
    df_features_encoded['Subscription Type'] = df_features_encoded['Subscription Type'].map(subscription_map)
    df_features_encoded['Contract Length'] = df_features_encoded['Contract Length'].map(contract_map)
    print("Mã hóa dữ liệu xong.")

    # 2C. VẼ BIỂU ĐỒ SAU MÃ HÓA 
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_features_encoded, x='Gender_Male', palette='ocean_r')
    plt.title('Phân bổ Giới tính (Sau mã hóa: 0=Female, 1=Male)')
    plt.savefig('static/eda_04_gender_encoded.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_features_encoded, x='Subscription Type', palette='cubehelix_r')
    plt.title('Phân bổ Loại hợp đồng (Sau mã hóa: 0=Basic, 1=Std, 2=Prem)')
    plt.savefig('static/eda_05_subscription_encoded.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_features_encoded, x='Contract Length', palette='nipy_spectral_r')
    plt.title('Phân bổ Độ dài hợp đồng (Sau mã hóa: 0=Monthly, 1=Q, 2=Annual)')
    plt.savefig('static/eda_06_contract_encoded.png')
    plt.close()

    # 2D. VẼ HEATMAP
    plt.figure(figsize=(12, 10))
    corr_matrix = df_features_encoded.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
    plt.title('Heatmap Ma trận tương quan (Sau khi mã hóa)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('static/eda_07_heatmap.png')
    plt.close()

    # 3. Lựa chọn thuộc tính
    X = df_features_encoded.drop(columns=['Churn'])
    y = df_features_encoded['Churn']
    feature_names = X.columns.tolist()

    # 4. CHUẨN HÓA (MINMAX)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # VẼ BIỂU ĐỒ So sánh Scaling
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.kdeplot(X['Age'], fill=True, label='Trước (Age)', color='red')
    sns.kdeplot(X['Total Spend'], fill=True, label='Trước (Total Spend)', color='blue')
    plt.title('Trước khi chuẩn hóa')
    plt.legend()
    plt.xlabel('Giá trị')
    plt.subplot(1, 2, 2)
    sns.kdeplot(df_scaled['Age'], fill=True, label='Sau (Age)', color='red')
    sns.kdeplot(df_scaled['Total Spend'], fill=True, label='Sau (Total Spend)', color='blue')
    plt.title('Sau khi dùng MinMaxScaler')
    plt.legend()
    plt.xlabel('Giá trị đã chuẩn hóa (0 đến 1)')
    plt.suptitle('So sánh dữ liệu Trước và Sau khi chuẩn hóa')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('static/eda_08_scaling_comparison.png')
    plt.close()
    print("Vẽ EDA hoàn tất.")

    # 5. CHIA DỮ LIỆU 
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, shuffle=True, random_state=42
    )
    # 6. HUẤN LUYỆN Rừng ngẫu nhiên
    model = RandomForestClassifier(
        random_state=42,
        max_depth=5,
        min_samples_leaf=10,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Huấn luyện mô hình xong.")
    
    # Tạo SHAP EXPLAINER
    explainer = shap.TreeExplainer(model, X_train)
    print("Đã tạo SHAP Explainer thành công.")

    # 7. ĐÁNH GIÁ MODEL VÀ VẼ BIỂU ĐỒ
    print("Đang vẽ biểu đồ Đánh giá Model...")
    
    rf_pred = model.predict(X_test)
    rf_prob = model.predict_proba(X_test)[:,1]
    
    # Classification Report
    print("\n" + "="*50)
    print("--- CLASSIFICATION REPORT (TỔNG QUAN) ---")
    report_text = classification_report(y_test, rf_pred, target_names=['Gắn bó (0)', 'Rời bỏ (1)'])
    print(report_text)
    print("="*50 + "\n")
    
    plt.figure(figsize=(8, 4))
    plt.text(0.01, 0.9, report_text, {'fontname': 'Courier New', 'fontsize': 12}, va='top', ha='left')
    plt.axis('off')
    plt.title('Báo cáo Phân loại (Classification Report)', fontsize=16)
    plt.tight_layout()
    plt.savefig('static/eval_01_classification_report.png')
    plt.close()

    # Confusion Matrix 
    print("Đang vẽ Confusion Matrix...")
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Gắn bó (0)', 'Rời bỏ (1)'],
                yticklabels=['Gắn bó (0)', 'Rời bỏ (1)'])
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.xlabel('Dự đoán (Predicted Label)')
    plt.ylabel('Thực tế (True Label)')
    plt.tight_layout()
    plt.savefig('static/eval_02_confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp_sorted = feat_imp.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=feat_imp_sorted, x='importance', y='feature', palette='vlag')
    plt.title('Độ quan trọng của thuộc tính (Random Forest)')
    plt.tight_layout()
    plt.savefig('static/eval_03_feature_importance.png')
    plt.close()

    # ROC Curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.figure(figsize=(8,6))
    plt.plot([0,1],[0,1],'k--', lw=2)
    plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC={auc_rf:.3f})', color='green')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Đường cong ROC (ROC Curve)")
    plt.legend(loc="lower right")
    plt.savefig('static/eval_04_roc_curve.png')
    plt.close()
    
    print("Vẽ và lưu biểu đồ đánh giá hoàn tất! ")
    
    return model, scaler, feature_names, explainer

# KHỞI TẠO CÁC BIẾN TOÀN CỤC 
print("Đang chuẩn bị server...")
model, scaler, feature_names, explainer = train_model_on_startup()
print("✅ Server sẵn sàng! Mô hình, Biểu đồ và SHAP Explainer đã được tạo.")


# KHỞI TẠO FLASK APP (SERVER) 
app = Flask(__name__)
CORS(app) 

# ENDPOINT /predict
@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        data = request.json
        print(f"Nhận được dữ liệu: {data}")

        # 1. Mã hóa input
        input_data_encoded = {
            "Age": int(data['age']),
            "Tenure": int(data['tenure']),
            "Usage Frequency": int(data['usage']),
            "Support Calls": int(data['support_calls']),
            "Payment Delay": int(data['payment_delay']),
            "Total Spend": float(data['total_spend']),
            "Last Interaction": int(data['last_interaction']),
            "Subscription Type": subscription_map[data['subscription']],
            "Contract Length": contract_map[data['contract_length']],
            "Gender_Male": 1 if data['gender'] == "Male" else 0
        }
        
        input_values = [input_data_encoded.get(f, 0) for f in feature_names]
        input_df_final = pd.DataFrame([input_values], columns=feature_names)
        input_scaled = scaler.transform(input_df_final)
        
        # 2. Dự đoán 
        prob_churn = model.predict_proba(input_scaled)[0, 1]
        prob_stay = 1 - prob_churn

        # 3. TÍNH TOÁN SHAP 
        print("Đang tính toán SHAP cho dự đoán...")
        shap_values_customer = explainer.shap_values(input_scaled)[0, :, 1]
        
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_customer
        })

        # Tách đặc trưng
        positive_factors = df_shap[df_shap['shap_value'] > 0].set_index('feature')['shap_value']
        negative_factors = df_shap[df_shap['shap_value'] < 0].set_index('feature')['shap_value'].abs()
        positive_factors = positive_factors.sort_values(ascending=False)
        negative_factors = negative_factors.sort_values(ascending=False)

        # 4. VẼ BIỂU ĐỒ PIE SHAP 
        fig_pos = create_pie_chart(positive_factors, '🔴 Đặc trưng làm khách hàng RỜI ĐI')
        fig_neg = create_pie_chart(negative_factors, '🟢 Đặc trưng GIỮ CHÂN khách hàng')
        
        # 5. Chuyển ảnh Matplotlib sang Base64
        def fig_to_base64(fig):
            img_buf = BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            plt.close(fig) # Đóng figure để giải phóng bộ nhớ
            return base64.b64encode(img_buf.getvalue()).decode('utf-8')

        pos_b64 = fig_to_base64(fig_pos)
        neg_b64 = fig_to_base64(fig_neg)
        print("Tạo ảnh SHAP Base64 thành công.")

        # 6. Gửi kết quả (BAO GỒM TIMESTAMP)
        current_time = datetime.now().strftime("%H:%M:%S - %d/%m/%Y") # <--- Lấy giờ hệ thống

        result = {
            'prob_churn': prob_churn,
            'prob_stay': prob_stay,
            'shap_pos_b64': pos_b64,
            'shap_neg_b64': neg_b64,
            'server_timestamp': current_time # <--- Trả về timestamp
        }
        return jsonify(result)
        
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return jsonify({'error': str(e)}), 500

# ENDPOINT ĐỂ PHỤC VỤ ẢNH 
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# CHẠY SERVER 
if __name__ == '__main__':
    import os
    # Render sẽ truyền một biến môi trường tên là PORT vào
    port = int(os.environ.get("PORT", 5000))
    
    print(f"Backend Server đang chạy tại cổng: {port}")
    # Host phải là 0.0.0.0 để server bên ngoài truy cập được vào
    app.run(host='0.0.0.0', port=port, debug=False)
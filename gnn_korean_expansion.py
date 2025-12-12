import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import random
import glob
from matplotlib.lines import Line2D # 범례 생성을 위해 추가

# ==========================================
# ⚙️ 설정
# ==========================================
GRAPH_PATH = "./outputs/graph/graph_data.pt"
ENCODER_PATH = "./outputs/graph/label_encoders.pt"
EMBEDDING_PATH = "./outputs/graph/dgl_node_embeddings_v3.pt"
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs/graph/gnn"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 전역 폰트 변수
GLOBAL_FONT_NAME = "sans-serif"

# ==========================================
# 🛠️ 유틸리티: 폰트 & 데이터 로드
# ==========================================
def init_font():
    """시각화용 다국어 폰트 설정"""
    global GLOBAL_FONT_NAME
    system_name = platform.system()
    
    if system_name == 'Windows':
        candidates = [
            ("c:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
            ("c:/Windows/Fonts/msyh.ttf", "Microsoft YaHei"),
        ]
    elif system_name == 'Darwin':
        candidates = [("/System/Library/Fonts/Supplemental/AppleGothic.ttf", "AppleGothic")]
    else:
        candidates = [("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic")]

    for fpath, fname in candidates:
        if os.path.exists(fpath):
            try:
                font_manager.fontManager.addfont(fpath)
                GLOBAL_FONT_NAME = font_manager.FontProperties(fname=fpath).get_name()
                rc('font', family=GLOBAL_FONT_NAME)
                print(f"🔤 시각화 폰트 로드: {GLOBAL_FONT_NAME}")
                break
            except: continue
    plt.rcParams['axes.unicode_minus'] = False

def load_resources():
    print("🔄 분석 리소스 로드 중...")
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(f"❌ 임베딩 파일이 없습니다.")

    try:
        data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
        encoders = torch.load(ENCODER_PATH, weights_only=False)
        embeddings = torch.load(EMBEDDING_PATH, map_location='cpu')
    except TypeError:
        data = torch.load(GRAPH_PATH, map_location='cpu')
        encoders = torch.load(ENCODER_PATH)
        embeddings = torch.load(EMBEDDING_PATH, map_location='cpu')
    
    print("✅ 데이터 로드 완료!")
    return data, encoders, embeddings

def get_korean_brands():
    """'한국_DATA.xlsx'를 읽어 한국 브랜드 목록을 추출합니다."""
    korean_file = os.path.join(DATA_DIR, "한국_DATA.xlsx")
    
    if not os.path.exists(korean_file):
        files = glob.glob(os.path.join(DATA_DIR, "*한국*.xlsx"))
        if not files:
            raise FileNotFoundError("❌ 한국 데이터 파일을 찾을 수 없습니다.")
        korean_file = files[0]
    
    print(f"🇰🇷 한국 데이터 로드 중: {os.path.basename(korean_file)}")
    df = pd.read_excel(korean_file)
    
    if '상표명칭' in df.columns: target_col = '상표명칭'
    elif '출원인' in df.columns: target_col = '출원인'
    else: target_col = df.columns[0]
    
    brands = df[target_col].dropna().astype(str).unique()
    return set(brands)

# ==========================================
# 🧠 AI 분석 엔진
# ==========================================
def get_top_korean_brands(data, encoders, top_k=5):
    """
    [선정 기준]
    한국 브랜드 중에서 '보유 상표 수(Degree)'가 가장 많은 상위 K개 기업을 선정합니다.
    이유: 데이터가 풍부할수록 GNN 예측의 신뢰도가 높기 때문입니다.
    """
    korean_brands_set = get_korean_brands()
    comp_names = encoders['company_classes']
    
    korean_indices = []
    for idx, name in enumerate(comp_names):
        if name in korean_brands_set:
            korean_indices.append(idx)
            
    if not korean_indices:
        print("❌ 매칭되는 한국 브랜드가 없습니다.")
        return [], []

    # 전체 기업의 상표 보유 수 계산
    edge_index = data['company', 'files', 'trademark'].edge_index
    all_degrees = torch.bincount(edge_index[0], minlength=len(comp_names))
    
    # 한국 브랜드만 필터링
    korean_degrees = all_degrees[korean_indices]
    
    # 상표 수 기준 내림차순 정렬하여 Top-K 추출
    top_vals, top_idx_local = torch.topk(korean_degrees, min(top_k, len(korean_indices)))
    top_indices_global = [korean_indices[i] for i in top_idx_local.tolist()]
    
    print(f"\n🏆 [Top {top_k} 한국 브랜드 선정 (기준: 상표 보유 수)]")
    for i, idx in enumerate(top_indices_global):
        print(f" {i+1}. {comp_names[idx]} (보유: {top_vals[i]}건)")
        
    return top_indices_global, top_vals.tolist()

def predict_expansion(data, encoders, embeddings, brand_idx, top_k=3):
    comp_emb = embeddings['company'][brand_idx]
    class_embs = embeddings['class']
    scores = torch.matmul(class_embs, comp_emb)
    
    edge_ct = data['company', 'files', 'trademark'].edge_index
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    
    my_tm_mask = (edge_ct[0] == brand_idx)
    my_tm_indices = edge_ct[1][my_tm_mask]
    
    my_class_mask = torch.isin(edge_tc[0], my_tm_indices)
    my_class_indices = edge_tc[1][my_class_mask]
    my_unique_classes = torch.unique(my_class_indices)
    
    scores[my_unique_classes] = -9999.0 # 이미 보유한 류 제외
    
    best_scores, best_indices = torch.topk(scores, top_k)
    
    recommendations = []
    class_names = encoders['class_classes']
    for idx, score in zip(best_indices, best_scores):
        recommendations.append((class_names[idx.item()], score.item()))
        
    return recommendations

# ==========================================
# 🎨 시각화 (범례 추가됨)
# ==========================================
def visualize_expansion(data, encoders, brand_name, recommendations, max_nodes=15):
    brand_idx = np.where(encoders['company_classes'] == brand_name)[0][0]
    
    # 데이터 준비
    edge_ct = data['company', 'files', 'trademark'].edge_index
    my_tm_indices = edge_ct[1][edge_ct[0] == brand_idx].tolist()
    if len(my_tm_indices) > max_nodes:
        my_tm_indices = random.sample(my_tm_indices, max_nodes)
        
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    tm_names = encoders['trademark_classes']
    class_names = encoders['class_classes']

    # 그래프 생성
    G = nx.Graph()
    G.add_node(brand_name, type='brand', size=2500, color='#FF6B6B')

    # 1. 현재 보유 (실선)
    for tm_idx in my_tm_indices:
        short_name = tm_names[tm_idx].split('_')[0][:6]
        tm_node = f"TM:{tm_idx}"
        
        G.add_node(tm_node, label=short_name, type='trademark', size=600, color='#4ECDC4')
        G.add_edge(brand_name, tm_node, style='solid', color='gray')

        mask_c = (edge_tc[0] == tm_idx)
        for c_idx in edge_tc[1][mask_c].tolist():
            c_name = class_names[c_idx]
            c_node = f"Class:{c_name}"
            if c_node not in G:
                G.add_node(c_node, label=f"{c_name}류", type='class', size=1200, color='#FFE66D')
            G.add_edge(tm_node, c_node, style='solid', color='gray')

    # 2. 미래 예측 (점선)
    for rank, (rec_class, score) in enumerate(recommendations):
        rec_node = f"Class:{rec_class}"
        if rec_node in G: continue
        
        label = f"★추천{rank+1}\n{rec_class}류"
        G.add_node(rec_node, label=label, type='recommendation', size=1500, color='#A8DADC')
        G.add_edge(brand_name, rec_node, style='dashed', color='#FF6B6B')

    # 그리기
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.9, seed=42)
    
    for n, d in G.nodes(data=True):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=d['color'], node_size=d['size'], alpha=0.9)
    
    edges = G.edges(data=True)
    solid = [(u,v) for u,v,d in edges if d.get('style')=='solid']
    dashed = [(u,v) for u,v,d in edges if d.get('style')=='dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid, width=1.0, edge_color='gray', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=dashed, width=2.5, edge_color='#FF6B6B', style='dashed')
    
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    labels[brand_name] = brand_name
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family=GLOBAL_FONT_NAME, font_weight='bold')
    
    # ---------------------------------------------------------
    # 📝 [추가됨] 범례 (Legend) 설정
    # ---------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Brand (분석 대상)', markerfacecolor='#FF6B6B', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Trademark (보유 상표)', markerfacecolor='#4ECDC4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Current Class (현재 사업)', markerfacecolor='#FFE66D', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='AI Recommendation (추천 신사업)', markerfacecolor='#A8DADC', markersize=15),
        Line2D([0], [0], color='gray', lw=1, label='Current Link (현황)'),
        Line2D([0], [0], color='#FF6B6B', lw=2, linestyle='--', label='AI Predicted Link (예측)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', prop={'size': 11, 'family': GLOBAL_FONT_NAME})

    plt.title(f"Korea Brand Expansion Prediction: {brand_name}", fontsize=16, fontfamily=GLOBAL_FONT_NAME)
    plt.axis('off')
    
    safe_name = "".join([c if c.isalnum() else "_" for c in brand_name])
    save_path = os.path.join(OUTPUT_DIR, f"KR_Expansion_{safe_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ 결과 저장: {save_path}")
    plt.close()

if __name__ == "__main__":
    init_font()
    data, encoders, embeddings = load_resources()
    
    # 1. 상위 5개 브랜드 선정
    top_indices, top_counts = get_top_korean_brands(data, encoders, top_k=5)
    
    print("\n🚀 [AI 예측 시작] 한국 상위 브랜드 신사업 확장 분석")
    for idx in top_indices:
        brand_name = encoders['company_classes'][idx]
        print(f"\n🏢 분석 중: {brand_name}...")
        
        recs = predict_expansion(data, encoders, embeddings, idx)
        for r_cls, r_score in recs:
            print(f"   👉 추천: {r_cls}류 (점수: {r_score:.2f})")
            
        visualize_expansion(data, encoders, brand_name, recs)
        
    print("\n✅ 모든 분석이 완료되었습니다. ./outputs/graph/gnn 폴더를 확인하세요.")
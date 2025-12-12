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
from matplotlib.lines import Line2D # 범례용 모듈

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
# 🛠️ 유틸리티
# ==========================================
def init_font():
    global GLOBAL_FONT_NAME
    system_name = platform.system()
    if system_name == 'Windows':
        candidates = [("c:/Windows/Fonts/malgun.ttf", "Malgun Gothic"), ("c:/Windows/Fonts/msyh.ttf", "Microsoft YaHei")]
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
    if not os.path.exists(EMBEDDING_PATH): raise FileNotFoundError(f"❌ 임베딩 파일이 없습니다.")
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
    korean_file = os.path.join(DATA_DIR, "한국_DATA.xlsx")
    if not os.path.exists(korean_file):
        files = glob.glob(os.path.join(DATA_DIR, "*한국*.xlsx"))
        if not files: return set()
        korean_file = files[0]
    
    df = pd.read_excel(korean_file)
    if '상표명칭' in df.columns: target_col = '상표명칭'
    elif '출원인' in df.columns: target_col = '출원인'
    else: target_col = df.columns[0]
    
    brands = df[target_col].dropna().astype(str).unique()
    return set(brands)

def get_top_korean_brands(data, encoders, top_k=5):
    """보유 상표 수가 많은 상위 K개 한국 브랜드 선정"""
    korean_brands_set = get_korean_brands()
    comp_names = encoders['company_classes']
    korean_indices = [i for i, name in enumerate(comp_names) if name in korean_brands_set]
    
    if not korean_indices: return []

    edge_index = data['company', 'files', 'trademark'].edge_index
    all_degrees = torch.bincount(edge_index[0], minlength=len(comp_names))
    korean_degrees = all_degrees[korean_indices]
    
    top_vals, top_idx_local = torch.topk(korean_degrees, min(top_k, len(korean_indices)))
    top_indices_global = [korean_indices[i] for i in top_idx_local.tolist()]
    
    return top_indices_global

# ==========================================
# 🧠 경쟁자 분석 엔진
# ==========================================
def find_competitors(encoders, embeddings, target_idx, top_k=5):
    target_emb = embeddings['company'][target_idx].unsqueeze(0)
    all_embs = embeddings['company']
    
    # 코사인 유사도
    sim_scores = F.cosine_similarity(target_emb, all_embs)
    sim_scores[target_idx] = -1.0 # 본인 제외
    
    best_scores, best_indices = torch.topk(sim_scores, top_k)
    
    competitors = []
    comp_names = encoders['company_classes']
    
    for idx, score in zip(best_indices, best_scores):
        competitors.append((comp_names[idx.item()], score.item(), idx.item()))
        
    return competitors

def get_shared_interests(data, encoders, idx1, idx2):
    """두 브랜드가 공통으로 보유한 류(Class) 찾기"""
    edge_ct = data['company', 'files', 'trademark'].edge_index
    tms1 = edge_ct[1][edge_ct[0] == idx1]
    tms2 = edge_ct[1][edge_ct[0] == idx2]
    
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    cls1 = edge_tc[1][torch.isin(edge_tc[0], tms1)].unique()
    cls2 = edge_tc[1][torch.isin(edge_tc[0], tms2)].unique()
    
    common_cls_ids = np.intersect1d(cls1.cpu().numpy(), cls2.cpu().numpy())
    class_names = encoders['class_classes']
    common_names = [class_names[i] for i in common_cls_ids]
    
    return common_names

# ==========================================
# 🎨 시각화 (범례 추가됨)
# ==========================================
def visualize_competitor_analysis(data, encoders, target_brand, competitors, target_idx):
    G = nx.Graph()
    G.add_node(target_brand, type='me', size=3000, color='#FF6B6B')
    
    print(f"\n🎨 경쟁사 관계도 생성 중...")
    
    for rank, (comp_name, score, comp_idx) in enumerate(competitors):
        # 경쟁사 노드 (유사도 표시)
        comp_node = f"{comp_name}\n({score:.2f})"
        G.add_node(comp_node, type='competitor', size=2000, color='#4ECDC4')
        
        # 공통 관심사(류) 찾기
        common_classes = get_shared_interests(data, encoders, target_idx, comp_idx)
        
        # 공통 류 연결 (최대 3개)
        for cls_name in common_classes[:3]:
            cls_node = f"Class:{cls_name}"
            
            if cls_node not in G:
                G.add_node(cls_node, label=f"{cls_name}류", type='shared', size=1200, color='#FFE66D')
                G.add_edge(target_brand, cls_node, style='solid', color='#FF6B6B')
            
            G.add_edge(comp_node, cls_node, style='solid', color='#4ECDC4')

    # 그리기
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    for n, d in G.nodes(data=True):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=d['color'], node_size=d['size'], alpha=0.9)
    
    edges = G.edges(data=True)
    solid_edges = [(u,v) for u,v,d in edges if d.get('style')=='solid']
    colors = [d['color'] for u,v,d in edges if d.get('style')=='solid']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2.0, edge_color=colors, alpha=0.6)
    
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family=GLOBAL_FONT_NAME, font_weight='bold')
    
    # ---------------------------------------------------------
    # 📝 [추가됨] 범례 (Legend) 설정
    # ---------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Brand (분석 대상)', markerfacecolor='#FF6B6B', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Competitor (유사 기업)', markerfacecolor='#4ECDC4', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Shared Interest (공통점)', markerfacecolor='#FFE66D', markersize=12),
        Line2D([0], [0], color='#FF6B6B', lw=2, label='My Link (보유)'),
        Line2D([0], [0], color='#4ECDC4', lw=2, label='Competitor Link (보유)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', prop={'size': 11, 'family': GLOBAL_FONT_NAME})

    plt.title(f"Competitor Analysis: {target_brand} (Top 5 Similar Brands)", fontsize=16, fontfamily=GLOBAL_FONT_NAME)
    plt.axis('off')
    
    safe_name = "".join([c if c.isalnum() else "_" for c in target_brand])
    save_path = os.path.join(OUTPUT_DIR, f"KR_Competitors_{safe_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ 결과 저장: {save_path}")
    plt.close()

# ==========================================
# 🚀 메인 실행
# ==========================================
if __name__ == "__main__":
    init_font()
    data, encoders, embeddings = load_resources()
    
    # 1. 상위 5개 한국 브랜드 선정
    top_indices = get_top_korean_brands(data, encoders, top_k=5)
    
    print("\n🚀 [AI 경쟁자 발굴 시작] 한국 상위 브랜드 유사도 분석")
    
    for idx in top_indices:
        brand_name = encoders['company_classes'][idx]
        print(f"\n🏢 분석 중: {brand_name}...")
        
        # 2. 경쟁자 탐색
        competitors = find_competitors(encoders, embeddings, idx, top_k=5)
        
        for name, score, _ in competitors:
            print(f"   🤜 유사 브랜드: {name:<20} (유사도: {score:.4f})")
            
        # 3. 시각화
        visualize_competitor_analysis(data, encoders, brand_name, competitors, idx)
        
    print("\n✅ 분석 완료. ./outputs/graph/gnn 폴더를 확인하세요.")
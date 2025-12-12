import torch
import torch.nn.functional as F
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import random

# ==========================================
# ⚙️ 설정
# ==========================================
# 파일 경로 (학습된 모델과 데이터)
GRAPH_PATH = "./outputs/graph/graph_data.pt"
ENCODER_PATH = "./outputs/graph/label_encoders.pt"
EMBEDDING_PATH = "./outputs/graph/dgl_node_embeddings_v3.pt"

# 결과 저장 경로
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
    
    font_files = []
    if system_name == 'Windows':
        candidates = [
            ("c:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
            ("c:/Windows/Fonts/msgothic.ttc", "MS Gothic"),
            ("c:/Windows/Fonts/msyh.ttf", "Microsoft YaHei")
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
        raise FileNotFoundError(f"❌ 임베딩 파일({EMBEDDING_PATH})이 없습니다. 학습(Training)을 먼저 완료하세요.")

    # 1. 원본 그래프 (연결 관계 확인용)
    try:
        data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
        encoders = torch.load(ENCODER_PATH, weights_only=False)
    except TypeError:
        data = torch.load(GRAPH_PATH, map_location='cpu')
        encoders = torch.load(ENCODER_PATH)

    # 2. 학습된 임베딩 (AI의 뇌)
    embeddings = torch.load(EMBEDDING_PATH, map_location='cpu')
    
    print("✅ 로드 완료!")
    return data, encoders, embeddings

# ==========================================
# 🧠 AI 분석 엔진
# ==========================================
def get_brand_index(encoders, brand_name):
    try:
        return np.where(encoders['company_classes'] == brand_name)[0][0]
    except IndexError:
        print(f"⚠️ 브랜드 '{brand_name}'을 데이터에서 찾을 수 없습니다.")
        return None

def analyze_ai_recommendations(data, encoders, embeddings, brand_idx, top_k=3):
    """
    [분석 1] GNN 기반 신사업(Class) 추천
    - 기업 벡터와 류(Class) 벡터의 내적(Dot Product) 점수가 높은 순으로 추천
    - 이미 진출한 분야는 제외
    """
    # 1. 임베딩 가져오기
    comp_emb = embeddings['company'][brand_idx] # [Hidden_Dim]
    class_embs = embeddings['class']            # [Num_Classes, Hidden_Dim]
    
    # 2. 예측 점수 계산 (내적)
    # 점수가 높을수록 AI가 "이 기업과 잘 맞는다"고 판단한 것
    scores = torch.matmul(class_embs, comp_emb)
    
    # 3. 이미 보유한 류 제외 (Masking)
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    edge_ct = data['company', 'files', 'trademark'].edge_index
    
    # 내 상표들 찾기
    my_tm_mask = (edge_ct[0] == brand_idx)
    my_tm_indices = edge_ct[1][my_tm_mask]
    
    # 내 상표들이 속한 류 찾기
    my_class_mask = torch.isin(edge_tc[0], my_tm_indices)
    my_class_indices = edge_tc[1][my_class_mask]
    my_unique_classes = torch.unique(my_class_indices)
    
    # 이미 가진 류는 점수 -무한대 처리
    scores[my_unique_classes] = -9999.0
    
    # 4. Top-K 추천
    best_scores, best_indices = torch.topk(scores, top_k)
    
    recommendations = []
    class_names = encoders['class_classes']
    
    print(f"\n🚀 [AI 추천] GNN이 예측한 진출 유망 분야 (Confidence Score)")
    print("-" * 50)
    for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
        cls_name = class_names[idx.item()]
        print(f" 🏆 {i+1}위: {cls_name}류 (예측점수: {score:.4f})")
        recommendations.append((cls_name, score.item()))
    
    return recommendations

def find_similar_brands(encoders, embeddings, brand_idx, top_k=3):
    """
    [분석 2] 유사 브랜드 탐색 (Competitor Analysis)
    - 임베딩 공간에서 코사인 유사도가 가장 높은 브랜드 찾기
    """
    comp_emb = embeddings['company'][brand_idx].unsqueeze(0) # [1, Dim]
    all_comp_embs = embeddings['company']                    # [N, Dim]
    
    # 코사인 유사도 계산
    sim_scores = F.cosine_similarity(comp_emb, all_comp_embs)
    
    # 자기 자신 제외하고 Top-K
    sim_scores[brand_idx] = -1.0
    best_scores, best_indices = torch.topk(sim_scores, top_k)
    
    comp_names = encoders['company_classes']
    
    print(f"\n🤝 [경쟁사 분석] 사업 구조가 가장 유사한 브랜드")
    print("-" * 50)
    for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
        similar_name = comp_names[idx.item()]
        print(f" 🥈 {i+1}위: {similar_name} (유사도: {score:.4f})")

# ==========================================
# 🎨 시각화 (현재 + 미래)
# ==========================================
def visualize_future_strategy(data, encoders, brand_name, recommendations, max_nodes=15):
    """
    현재 보유한 상표/류(실선)와 AI가 추천한 미래 전략(점선)을 시각화
    """
    brand_idx = get_brand_index(encoders, brand_name)
    if brand_idx is None: return

    comp_names = encoders['company_classes']
    tm_names = encoders['trademark_classes']
    class_names = encoders['class_classes']

    # 그래프 생성
    G = nx.Graph()
    G.add_node(brand_name, type='brand', size=2500, color='#FF6B6B') # 메인 브랜드

    # 1. 현재 상태 그리기 (실선)
    edge_ct = data['company', 'files', 'trademark'].edge_index
    my_tm_indices = edge_ct[1][edge_ct[0] == brand_idx].tolist()
    
    # 상표 샘플링
    if len(my_tm_indices) > max_nodes:
        my_tm_indices = random.sample(my_tm_indices, max_nodes)
        
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index

    # 기존 상표 및 류 연결
    existing_classes = set()
    for tm_idx in my_tm_indices:
        # 상표 노드
        raw_name = tm_names[tm_idx]
        short_name = raw_name.split('_')[0][:6]
        tm_node = f"TM:{tm_idx}"
        
        G.add_node(tm_node, label=short_name, type='trademark', size=600, color='#4ECDC4')
        G.add_edge(brand_name, tm_node, style='solid', color='gray', weight=1)

        # 류 연결
        mask_c = (edge_tc[0] == tm_idx)
        for c_idx in edge_tc[1][mask_c].tolist():
            c_name = class_names[c_idx]
            c_node = f"Class:{c_name}"
            
            if c_node not in G:
                G.add_node(c_node, label=f"{c_name}류", type='class', size=1200, color='#FFE66D') # 노랑
                existing_classes.add(c_name)
            
            G.add_edge(tm_node, c_node, style='solid', color='gray', weight=1)

    # 2. AI 추천(미래) 그리기 (점선)
    print("\n🎨 미래 전략지도 생성 중...")
    for rank, (rec_class, score) in enumerate(recommendations):
        rec_node = f"Class:{rec_class}"
        
        # 이미 노드가 있다면(기존 보유) 패스 (하지만 로직상 없어야 함)
        if rec_node in G: continue
        
        # 추천 노드 추가 (색상을 다르게)
        label = f"★추천{rank+1}\n{rec_class}류"
        G.add_node(rec_node, label=label, type='recommendation', size=1500, color='#A8DADC') # 하늘색
        
        # 브랜드와 직접 점선 연결
        G.add_edge(brand_name, rec_node, style='dashed', color='#FF6B6B', weight=2)

    # 3. 그리기 설정
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    # 노드 그리기
    for n, d in G.nodes(data=True):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=d['color'], node_size=d['size'], alpha=0.9)
    
    # 엣지 그리기 (스타일 구분)
    edges = G.edges(data=True)
    solid_edges = [(u, v) for u, v, d in edges if d.get('style') == 'solid']
    dashed_edges = [(u, v) for u, v, d in edges if d.get('style') == 'dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=1.0, edge_color='gray', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=2.5, edge_color='#FF6B6B', style='dashed', alpha=0.8)

    # 라벨
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    labels[brand_name] = brand_name
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family=GLOBAL_FONT_NAME, font_weight='bold')

    # 범례
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Brand (현재)', markerfacecolor='#FF6B6B', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Trademark (상표)', markerfacecolor='#4ECDC4', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Current Class (진출함)', markerfacecolor='#FFE66D', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='AI Recommendation (유망)', markerfacecolor='#A8DADC', markersize=15),
        plt.Line2D([0], [0], color='#FF6B6B', lw=2, linestyle='--', label='Predicted Link')
    ]
    plt.legend(handles=legend_elements, loc='upper left', prop={'size': 11, 'family': GLOBAL_FONT_NAME})

    plt.title(f"AI Brand Expansion Strategy: {brand_name}", fontsize=16, fontfamily=GLOBAL_FONT_NAME)
    plt.axis('off')
    
    # 저장
    safe_name = "".join([c if c.isalnum() else "_" for c in brand_name])
    save_path = os.path.join(OUTPUT_DIR, f"gnn_strategy_{safe_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 전략 지도 저장 완료: {save_path}")
    plt.show()

# ==========================================
# 🚀 메인 실행
# ==========================================
if __name__ == "__main__":
    init_font()
    data, encoders, embeddings = load_resources()
    
    # [입력] 분석할 브랜드 이름 (보유 상표 수 1위 자동 선택)
    edge_index = data['company', 'files', 'trademark'].edge_index
    top_idx = torch.bincount(edge_index[0]).argmax().item()
    target_brand = encoders['company_classes'][top_idx]
    
    # target_brand = "SAMSUNG" # 직접 입력 가능
    
    print(f"\n🎯 분석 대상 브랜드: {target_brand}")

    # 1. AI 추천 (Class)
    brand_idx = get_brand_index(encoders, target_brand)
    if brand_idx is not None:
        recs = analyze_ai_recommendations(data, encoders, embeddings, brand_idx)
        
        # 2. 유사 브랜드 분석
        find_similar_brands(encoders, embeddings, brand_idx)
        
        # 3. 전략 지도 시각화
        visualize_future_strategy(data, encoders, target_brand, recs)
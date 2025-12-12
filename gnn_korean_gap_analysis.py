import torch
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from matplotlib.lines import Line2D
import platform
import random
import glob

# ==========================================
# ⚙️ 설정
# ==========================================
GRAPH_PATH = "./outputs/graph/graph_data.pt"
ENCODER_PATH = "./outputs/graph/label_encoders.pt"
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
    try:
        data = torch.load(GRAPH_PATH, map_location='cpu', weights_only=False)
        encoders = torch.load(ENCODER_PATH, weights_only=False)
    except TypeError:
        data = torch.load(GRAPH_PATH, map_location='cpu')
        encoders = torch.load(ENCODER_PATH)
    print("✅ 데이터 로드 완료!")
    return data, encoders

def get_korean_brands():
    korean_file = os.path.join(DATA_DIR, "한국_DATA.xlsx")
    if not os.path.exists(korean_file):
        files = glob.glob(os.path.join(DATA_DIR, "*한국*.xlsx"))
        if not files: return set()
        korean_file = files[0]
    
    df = pd.read_excel(korean_file)
    target_col = '상표명칭' if '상표명칭' in df.columns else df.columns[0]
    brands = df[target_col].dropna().astype(str).unique()
    return set(brands)

# ==========================================
# 🧠 [NEW] 다양성 기반 브랜드 선정
# ==========================================
def get_diverse_top_korean_brands(data, encoders, top_k=5):
    """
    [선정 기준 변경]
    단순히 전체 1~5등을 뽑는 게 아니라,
    '주요 산업군(Class)' 별로 1등 브랜드를 하나씩 뽑습니다.
    (예: 전자 1등, 화장품 1등, 식품 1등, 패션 1등...)
    """
    print("\n🔍 한국 브랜드 산업별 대표주자 선별 중...")
    
    korean_brands_set = get_korean_brands()
    comp_names = encoders['company_classes']
    class_names = encoders['class_classes']
    
    # 한국 브랜드 인덱스 필터링
    korean_indices = [i for i, name in enumerate(comp_names) if name in korean_brands_set]
    if not korean_indices: return []

    # 1. 브랜드별 주력 Class 계산 (Sparse Matrix 활용)
    # (Brand -> Trademark) * (Trademark -> Class) = (Brand -> Class Count)
    
    # 행렬 A: Brand -> Trademark
    edge_ct = data['company', 'files', 'trademark'].edge_index
    n_comp = len(comp_names)
    n_tm = data['trademark'].num_nodes
    
    # 텐서 생성
    indices_ct = edge_ct
    values_ct = torch.ones(edge_ct.size(1))
    adj_ct = torch.sparse_coo_tensor(indices_ct, values_ct, (n_comp, n_tm))
    
    # 행렬 B: Trademark -> Class
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    n_class = len(class_names)
    
    indices_tc = edge_tc
    values_tc = torch.ones(edge_tc.size(1))
    adj_tc = torch.sparse_coo_tensor(indices_tc, values_tc, (n_tm, n_class))
    
    # 행렬 곱 (Brand x Class)
    adj_cc = torch.sparse.mm(adj_ct, adj_tc).to_dense() # [Num_Brand, Num_Class]
    
    # 2. 한국 브랜드 데이터만 추출
    korean_stats = adj_cc[korean_indices] # [Num_Korean_Brands, Num_Class]
    
    # 3. 각 브랜드의 Main Class와 총 상표 수 확인
    brand_main_class_ids = torch.argmax(korean_stats, dim=1) # 각 브랜드의 주력 류 ID
    brand_total_counts = torch.sum(korean_stats, dim=1)      # 각 브랜드의 총 상표 수
    
    # 4. 산업군(Class)별로 그룹핑하여 1등 뽑기
    # Class별로 (총 상표 수, 브랜드 로컬 인덱스) 리스트 생성
    class_leaders = {}
    
    for local_idx, (class_id, count) in enumerate(zip(brand_main_class_ids, brand_total_counts)):
        c_id = class_id.item()
        cnt = count.item()
        if cnt == 0: continue
        
        if c_id not in class_leaders:
            class_leaders[c_id] = []
        class_leaders[c_id].append((cnt, local_idx))
    
    # 5. 가장 인기 있는 산업군(Class) Top-K 선정 (브랜드가 많이 몰린 류 순서)
    # class_leaders의 길이(해당 류를 주력으로 하는 브랜드 수)로 정렬
    popular_classes = sorted(class_leaders.keys(), key=lambda k: len(class_leaders[k]), reverse=True)[:top_k]
    
    final_indices = []
    
    print(f"\n🏆 [다양성 기준 Top {top_k} 선정] 각 산업군 별 1위 브랜드")
    for c_id in popular_classes:
        # 해당 Class 내에서 상표 수가 가장 많은 브랜드 1개 선정
        leaders = sorted(class_leaders[c_id], key=lambda x: x[0], reverse=True)
        top_brand_local_idx = leaders[0][1]
        top_brand_count = leaders[0][0]
        
        # 전체 인덱스로 변환
        global_idx = korean_indices[top_brand_local_idx]
        brand_name = comp_names[global_idx]
        class_name = class_names[c_id]
        
        print(f" - [{class_name}류 1위] {brand_name} (보유: {int(top_brand_count)}건)")
        final_indices.append(global_idx)
        
    return final_indices

# ==========================================
# 🧠 갭 분석 (Gap Analysis) 엔진 (버그 수정됨)
# ==========================================
def analyze_gap_strategy(data, encoders, brand_idx, top_k=5):
    comp_names = encoders['company_classes']
    class_names = encoders['class_classes']
    group_names = encoders['group_classes']
    brand_name = comp_names[brand_idx]

    # 1. 내 상표 찾기
    edge_ct = data['company', 'files', 'trademark'].edge_index
    my_tm_mask = (edge_ct[0] == brand_idx)
    my_tm_indices = edge_ct[1][my_tm_mask]

    if len(my_tm_indices) == 0: return None

    # 2. 나의 주력 류(Class) 찾기
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    my_cls_mask = torch.isin(edge_tc[0], my_tm_indices)
    my_cls_indices = edge_tc[1][my_cls_mask]
    
    if len(my_cls_indices) == 0: return None
    main_class_idx = torch.mode(my_cls_indices).values.item()
    main_class_name = class_names[main_class_idx]
    
    print(f"\n🏢 [{brand_name}]의 주력 사업: {main_class_name}류")

    # 3. 주력 류에 속한 전체 상표 찾기
    class_tm_mask = (edge_tc[1] == main_class_idx)
    global_tm_indices = edge_tc[0][class_tm_mask]
    
    # 4. 전체 시장 유사군 통계
    edge_tg = data['trademark', 'has_code', 'group'].edge_index
    global_group_mask = torch.isin(edge_tg[0], global_tm_indices)
    global_group_indices = edge_tg[1][global_group_mask]
    global_group_counts = torch.bincount(global_group_indices, minlength=len(group_names))
    
    # 5. 내가 가진 유사군 찾기
    my_group_mask = torch.isin(edge_tg[0], my_tm_indices)
    my_group_indices = edge_tg[1][my_group_mask]
    my_unique_groups = torch.unique(my_group_indices)
    
    # 6. 갭 계산
    candidates = global_group_counts.clone()
    candidates[my_unique_groups] = -1 # 이미 가진건 제외
    
    gap_vals, gap_indices = torch.topk(candidates, top_k)
    
    gaps = []
    print(f" 🚨 [경고] 경쟁사들은 확보했지만 귀사는 누락된 핵심 유사군 (Top {top_k})")
    for idx, count in zip(gap_indices, gap_vals):
        if count.item() <= 0: continue
        g_name = group_names[idx.item()]
        print(f"   👉 누락됨: {g_name} (시장 출원 수: {count.item()}건)")
        gaps.append(g_name)
        
    # 7. [수정됨] 내가 잘하고 있는 유사군 (안전하게 Zip 사용)
    my_counts = torch.bincount(my_group_indices, minlength=len(group_names))
    # 내가 가진 것 중 Top 3
    my_strong_indices = torch.argsort(my_counts, descending=True)[:3]
    
    my_strong_groups = []
    for idx in my_strong_indices:
        count = my_counts[idx].item()
        if count > 0:
            my_strong_groups.append(group_names[idx.item()])
    
    return {
        'main_class': main_class_name,
        'gaps': gaps,
        'my_strong': my_strong_groups
    }

# ==========================================
# 🎨 시각화
# ==========================================
def visualize_gap_analysis(brand_name, analysis_result):
    if not analysis_result: return

    main_class = analysis_result['main_class']
    gaps = analysis_result['gaps']
    my_strong = analysis_result['my_strong']

    G = nx.Graph()
    center_node = f"{main_class}류\n(주력시장)"
    G.add_node(center_node, type='class', size=3000, color='#FFE66D')
    
    G.add_node(brand_name, type='me', size=2500, color='#FF6B6B')
    G.add_edge(brand_name, center_node, style='solid')
    
    # Safe Zone
    for g_name in my_strong:
        node_id = f"{g_name}\n(보유)"
        G.add_node(node_id, type='safe', size=1500, color='#4ECDC4')
        G.add_edge(center_node, node_id, style='solid', color='gray')
        G.add_edge(brand_name, node_id, style='solid', color='gray')

    # Gap Zone
    for g_name in gaps:
        node_id = f"{g_name}\n(누락!)"
        G.add_node(node_id, type='gap', size=1800, color='#FF9F1C')
        G.add_edge(center_node, node_id, style='dashed', color='#FF9F1C')

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.7, seed=42)
    
    for n, d in G.nodes(data=True):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=d['color'], node_size=d['size'], alpha=0.9)
    
    edges = G.edges(data=True)
    solid = [(u,v) for u,v,d in edges if d.get('style')=='solid']
    dashed = [(u,v) for u,v,d in edges if d.get('style')=='dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid, width=1.5, edge_color='gray', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=dashed, width=2.5, edge_color='#FF9F1C', style='dashed')
    
    labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family=GLOBAL_FONT_NAME, font_weight='bold')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Brand (나)', markerfacecolor='#FF6B6B', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Main Market (주력 시장)', markerfacecolor='#FFE66D', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Safe Zone (이미 확보함)', markerfacecolor='#4ECDC4', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='GAP / RISK (누락된 유사군)', markerfacecolor='#FF9F1C', markersize=15),
        Line2D([0], [0], color='gray', lw=1, label='Existing Link'),
        Line2D([0], [0], color='#FF9F1C', lw=2, linestyle='--', label='Market Trend (나는 없음)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', prop={'size': 11, 'family': GLOBAL_FONT_NAME})

    plt.title(f"Defensive Strategy: {brand_name} (Gap Analysis)", fontsize=16, fontfamily=GLOBAL_FONT_NAME)
    plt.axis('off')
    
    safe_name = "".join([c if c.isalnum() else "_" for c in brand_name])
    save_path = os.path.join(OUTPUT_DIR, f"KR_GapAnalysis_{safe_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ 방어 전략 지도 저장: {save_path}")
    plt.close()

if __name__ == "__main__":
    init_font()
    data, encoders = load_resources()
    
    # [변경된 함수 호출]
    top_indices = get_diverse_top_korean_brands(data, encoders, top_k=5)
    
    print("\n🚀 [AI 방어 전략 수립] 갭 분석(Gap Analysis) 시작")
    
    for idx in top_indices:
        brand_name = encoders['company_classes'][idx]
        
        # 2. 갭 분석 실행
        result = analyze_gap_strategy(data, encoders, idx, top_k=5)
        
        # 3. 시각화
        visualize_gap_analysis(brand_name, result)
        
    print("\n✅ 모든 분석 완료. ./outputs/graph/gnn 폴더를 확인하세요.")
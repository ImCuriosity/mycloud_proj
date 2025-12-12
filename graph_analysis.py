import torch
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import random
from matplotlib.lines import Line2D  # 💡 범례 생성을 위한 모듈 추가

# ==========================================
# ⚙️ 설정
# ==========================================
GRAPH_PATH = "./outputs/graph/graph_data.pt"
ENCODER_PATH = "./outputs/graph/label_encoders.pt"

# 전역 폰트 변수
GLOBAL_FONT_NAME = "sans-serif"

# ==========================================
# 🛠️ 유틸리티: 폰트 & 데이터 로드
# ==========================================
def init_font():
    """다국어 폰트 설정 (시각화용)"""
    global GLOBAL_FONT_NAME
    system_name = platform.system()
    
    if system_name == 'Windows':
        candidates = [
            ("c:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
            ("c:/Windows/Fonts/msyh.ttf", "Microsoft YaHei"),
            ("c:/Windows/Fonts/msgothic.ttc", "MS Gothic")
        ]
    elif system_name == 'Darwin':
        candidates = [("/System/Library/Fonts/Supplemental/AppleGothic.ttf", "AppleGothic")]
    else:
        candidates = [("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic")]

    for fpath, fname in candidates:
        if os.path.exists(fpath):
            try:
                font_manager.fontManager.addfont(fpath)
                prop = font_manager.FontProperties(fname=fpath)
                GLOBAL_FONT_NAME = prop.get_name()
                rc('font', family=GLOBAL_FONT_NAME)
                print(f"🔤 시각화 폰트 로드: {GLOBAL_FONT_NAME}")
                break
            except: continue
    plt.rcParams['axes.unicode_minus'] = False

def load_data():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("❌ 그래프 데이터가 없습니다. graph_generator.py를 먼저 실행하세요.")
    
    print("🔄 데이터 로드 중...")
    try:
        data = torch.load(GRAPH_PATH, weights_only=False)
        encoders = torch.load(ENCODER_PATH, weights_only=False)
    except TypeError:
        data = torch.load(GRAPH_PATH)
        encoders = torch.load(ENCODER_PATH)
    
    print("✅ 데이터 로드 완료.")
    return data, encoders

# ==========================================
# 📊 분석 엔진
# ==========================================
def analyze_brand_stats(data, encoders, target_brand_name):
    """
    특정 브랜드의 보유 상표, 주력 류, 주력 유사군을 분석합니다.
    """
    comp_names = encoders['company_classes']
    class_names = encoders['class_classes']
    group_names = encoders['group_classes']

    # 1. 브랜드 인덱스 찾기
    try:
        brand_idx = np.where(comp_names == target_brand_name)[0][0]
    except IndexError:
        print(f"⚠️ 브랜드 '{target_brand_name}'을 찾을 수 없습니다.")
        return None

    # 2. 보유 상표(Trademark) 찾기
    edge_ct = data['company', 'files', 'trademark'].edge_index
    mask_t = (edge_ct[0] == brand_idx)
    my_tm_indices = edge_ct[1][mask_t] # Tensor
    
    num_tms = len(my_tm_indices)
    
    if num_tms == 0:
        print(f"⚠️ '{target_brand_name}'은 보유한 상표가 없습니다.")
        return None

    # 3. 주력 류(Class) 분석
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    # 내 상표들이 가리키는 류 찾기 (isin 활용)
    mask_c = torch.isin(edge_tc[0], my_tm_indices)
    my_class_indices = edge_tc[1][mask_c]
    
    # 카운팅
    cls_ids, cls_counts = torch.unique(my_class_indices, return_counts=True)
    top_c_k = min(3, len(cls_ids))
    if top_c_k > 0:
        top_c_val, top_c_idx = torch.topk(cls_counts, top_c_k)
        top_classes = [class_names[cls_ids[i]] for i in top_c_idx]
    else:
        top_classes = []

    # 4. 주력 유사군(Group) 분석 (★ 핵심 추가)
    edge_tg = data['trademark', 'has_code', 'group'].edge_index
    mask_g = torch.isin(edge_tg[0], my_tm_indices)
    my_group_indices = edge_tg[1][mask_g]
    
    # 카운팅
    grp_ids, grp_counts = torch.unique(my_group_indices, return_counts=True)
    top_g_k = min(5, len(grp_ids))
    if top_g_k > 0:
        top_g_val, top_g_idx = torch.topk(grp_counts, top_g_k)
        top_groups = [group_names[grp_ids[i]] for i in top_g_idx]
    else:
        top_groups = []

    # 결과 출력
    print(f"\n🏢 [브랜드 분석] {target_brand_name}")
    print("-" * 40)
    print(f" 📌 보유 상표 수 : {num_tms}건")
    print(f" 📌 주력 류(Class): {', '.join(top_classes)} (Top 3)")
    print(f" 📌 주력 유사군   : {', '.join(top_groups)} (Top 5)")
    
    return {
        'brand_idx': brand_idx,
        'my_tm_indices': my_tm_indices,
        'my_class_indices': my_class_indices,
        'my_group_indices': my_group_indices
    }

def recommend_gap_analysis(data, encoders, brand_stats):
    """
    [Gap Analysis]
    시장 전체 트렌드와 비교하여, 이 브랜드가 놓치고 있는 '유망 유사군'을 추천합니다.
    """
    if brand_stats is None: return

    print("\n🚀 [AI 추천] 브랜드 확장 기회 (Gap Analysis)")
    print(" 👉 경쟁 브랜드들은 확보했지만, 귀사는 아직 없는 '알짜배기' 영역입니다.")
    print("-" * 60)

    class_names = encoders['class_classes']
    group_names = encoders['group_classes']

    # 1. 전체 유사군 인기 순위 계산 (Market Trend)
    edge_tg = data['trademark', 'has_code', 'group'].edge_index
    global_group_counts = torch.bincount(edge_tg[1], minlength=data['group'].num_nodes)

    # 2. 이미 보유한 유사군은 제외 (Masking)
    my_groups = brand_stats['my_group_indices']
    unique_my_groups = torch.unique(my_groups)
    
    candidates = global_group_counts.clone()
    candidates[unique_my_groups] = -1 # 보유한건 점수 삭제

    # 3. Top-K 추천
    top_k = 5
    rec_vals, rec_indices = torch.topk(candidates, top_k)
    
    for i, (idx, count) in enumerate(zip(rec_indices, rec_vals)):
        if count == -1: continue
        g_name = group_names[idx.item()]
        
        # 해당 유사군이 속한 대표 류 찾기 (역추적)
        # (간단히 그래프에서 해당 그룹과 연결된 상표 하나를 찾아 그 상표의 류를 확인)
        sample_tm_mask = (edge_tg[1] == idx)
        if sample_tm_mask.any():
            sample_tm = edge_tg[0][sample_tm_mask][0]
            # TM -> Class
            edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
            tm_c_mask = (edge_tc[0] == sample_tm)
            if tm_c_mask.any():
                c_idx = edge_tc[1][tm_c_mask][0]
                c_name = class_names[c_idx.item()]
            else:
                c_name = "?"
        else:
            c_name = "?"

        print(f" 🏆 {i+1}순위: 유사군 [{g_name:<7}] (관련 류: {c_name}류) - 시장 점유 {count.item()}건")

# ==========================================
# 🎨 시각화 엔진 (통합됨)
# ==========================================
def visualize_brand(data, encoders, target_brand, max_nodes=20):
    """분석된 브랜드의 그래프를 그립니다."""
    comp_names = encoders['company_classes']
    tm_names = encoders['trademark_classes']
    class_names = encoders['class_classes']
    group_names = encoders['group_classes']

    try:
        target_idx = np.where(comp_names == target_brand)[0][0]
    except: return

    # 연결 데이터 추출
    edge_ct = data['company', 'files', 'trademark'].edge_index
    my_tm_indices = edge_ct[1][edge_ct[0] == target_idx].tolist()
    
    # 샘플링
    if len(my_tm_indices) > max_nodes:
        my_tm_indices = random.sample(my_tm_indices, max_nodes)

    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    edge_tg = data['trademark', 'has_code', 'group'].edge_index

    G = nx.Graph()
    G.add_node(target_brand, type='brand', size=2500, color='#FF6B6B')

    # 노드 및 엣지 추가
    for tm_idx in my_tm_indices:
        # 상표
        raw_name = tm_names[tm_idx]
        short_name = raw_name.split('_')[0][:8]
        tm_node = f"TM:{tm_idx}"
        G.add_node(tm_node, label=short_name, type='trademark', size=800, color='#4ECDC4')
        G.add_edge(target_brand, tm_node)

        # 류 (Class)
        mask_c = (edge_tc[0] == tm_idx)
        for c_idx in edge_tc[1][mask_c].tolist():
            c_name = class_names[c_idx]
            c_node = f"Class:{c_name}"
            if not G.has_node(c_node):
                G.add_node(c_node, label=f"{c_name}류", type='class', size=1200, color='#FFE66D')
            G.add_edge(tm_node, c_node)

        # 유사군 (Group)
        mask_g = (edge_tg[0] == tm_idx)
        for g_idx in edge_tg[1][mask_g].tolist():
            g_name = group_names[g_idx]
            g_node = f"Group:{g_name}"
            if not G.has_node(g_node):
                G.add_node(g_node, label=g_name, type='group', size=1000, color='#1A535C')
            G.add_edge(tm_node, g_node)

    # 그리기
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=0.6)
    
    # 타입별 그리기
    for n, d in G.nodes(data=True):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=d['color'], node_size=d['size'], alpha=0.9)
    
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')
    
    # 라벨 (폰트 주입)
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    # 타겟 브랜드는 그대로 출력, 나머지는 그대로
    labels[target_brand] = target_brand
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family=GLOBAL_FONT_NAME)
    
    # ---------------------------------------------------------
    # 📝 [추가됨] 범례 (Legend) 설정
    # ---------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Brand (분석 대상)', markerfacecolor='#FF6B6B', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Trademark (상표)', markerfacecolor='#4ECDC4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Class (류 - 산업군)', markerfacecolor='#FFE66D', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Group (유사군 - 세부품목)', markerfacecolor='#1A535C', markersize=12)
    ]
    plt.legend(handles=legend_elements, loc='upper left', prop={'size': 11, 'family': GLOBAL_FONT_NAME})

    # 타이틀 & 저장
    plt.title(f"Brand Ecosystem: {target_brand}", fontsize=15, fontfamily=GLOBAL_FONT_NAME)
    plt.axis('off')
    
    safe_name = "".join([c if c.isalnum() else "_" for c in target_brand])
    save_path = f"./outputs/graph/analysis_{safe_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ 시각화 저장 완료: {save_path}")
    plt.show()

# ==========================================
# 🚀 메인 실행
# ==========================================
if __name__ == "__main__":
    init_font()
    data, encoders = load_data()
    
    # [입력] 분석하고 싶은 브랜드 이름 (보유 상표 수 1위 자동 선택)
    edge_index = data['company', 'files', 'trademark'].edge_index
    top_idx = torch.bincount(edge_index[0]).argmax().item()
    target_brand = encoders['company_classes'][top_idx]
    
    # 직접 입력하려면 아래 주석 해제
    # target_brand = "SAMSUNG" 
    
    # 1. 통계 분석
    stats = analyze_brand_stats(data, encoders, target_brand)
    
    # 2. 갭 분석 (추천)
    if stats:
        recommend_gap_analysis(data, encoders, stats)
        
        # 3. 시각화
        print("\n🎨 그래프 생성 중...")
        visualize_brand(data, encoders, target_brand)
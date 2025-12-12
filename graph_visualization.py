import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
import random
import numpy as np
import platform

# ==========================================
# ⚙️ 설정
# ==========================================
GRAPH_PATH = "./outputs/graph/pet_graph_data.pt"
ENCODER_PATH = "./outputs/graph/label_encoders.pt"

# 전역 폰트 변수
CHOSEN_FONT = "sans-serif"

def init_font():
    """
    가장 강력한 다국어 폰트 1개를 선정하여 등록하고 그 이름을 반환합니다.
    (NetworkX에 직접 전달하기 위함)
    """
    global CHOSEN_FONT
    system_name = platform.system()
    
    # 후보군: (파일경로, 폰트이름)
    # Microsoft YaHei: 중국어/한국어/일본어/영어 커버리지 우수
    # Malgun Gothic: 한국어 최적화 (일부 중국어 깨짐)
    candidates = []
    
    if system_name == 'Windows':
        candidates = [
            ("c:/Windows/Fonts/msyh.ttf", "Microsoft YaHei"),   # 1순위 (다국어 최강)
            ("c:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),   # 2순위
            ("c:/Windows/Fonts/msgothic.ttc", "MS Gothic"),     # 3순위
        ]
    elif system_name == 'Darwin':
        candidates = [
            ("/System/Library/Fonts/PingFang.ttc", "PingFang SC"),
            ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", "AppleGothic")
        ]
    else:
        candidates = [
            ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK JP")
        ]

    # 사용 가능한 첫 번째 폰트 찾기
    for fpath, fname in candidates:
        if os.path.exists(fpath):
            try:
                # 1. 폰트 파일 등록
                font_manager.fontManager.addfont(fpath)
                
                # 2. 전역 설정 (제목/범례용)
                rc('font', family=fname)
                
                # 3. NetworkX 전달용 변수 저장
                CHOSEN_FONT = fname
                print(f"✅ 폰트 설정 완료: '{CHOSEN_FONT}' (파일: {fpath})")
                return
            except Exception as e:
                print(f"⚠️ 폰트 등록 실패 ({fname}): {e}")
    
    print("⚠️ 적절한 폰트를 찾지 못해 시스템 기본값을 사용합니다.")

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

def load_data():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("그래프 파일이 없습니다.")
    
    print("🔄 데이터 로드 중...")
    try:
        data = torch.load(GRAPH_PATH, weights_only=False)
        encoders = torch.load(ENCODER_PATH, weights_only=False)
    except TypeError:
        data = torch.load(GRAPH_PATH)
        encoders = torch.load(ENCODER_PATH)
        
    return data, encoders

def visualize_ego_graph(data, encoders, target_brand=None, max_nodes=20):
    comp_names = encoders['company_classes']
    tm_names = encoders['trademark_classes']
    class_names = encoders['class_classes']
    group_names = encoders['group_classes']

    # 타겟 브랜드 선택
    if target_brand is None:
        rand_idx = random.randint(0, data['company'].num_nodes - 1)
        target_name = comp_names[rand_idx]
        target_idx = rand_idx
    else:
        try:
            target_idx = np.where(comp_names == target_brand)[0][0]
            target_name = target_brand
        except IndexError:
            print(f"❌ '{target_brand}' 브랜드를 찾을 수 없습니다.")
            return

    print(f"🎨 '{target_name}' 브랜드의 연결 관계를 시각화합니다...")

    # 연결된 노드 찾기
    edge_ct = data['company', 'files', 'trademark'].edge_index
    mask = (edge_ct[0] == target_idx)
    my_tm_indices = edge_ct[1][mask].tolist()
    
    if len(my_tm_indices) > max_nodes:
        print(f"   ℹ️ 상표가 너무 많아({len(my_tm_indices)}개), {max_nodes}개만 임의로 표시합니다.")
        my_tm_indices = random.sample(my_tm_indices, max_nodes)

    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    edge_tg = data['trademark', 'has_code', 'group'].edge_index
    
    related_classes = set()
    related_groups = set()
    valid_edges_tc = []
    valid_edges_tg = []

    for tm_idx in my_tm_indices:
        mask_c = (edge_tc[0] == tm_idx)
        c_indices = edge_tc[1][mask_c].tolist()
        for c_idx in c_indices:
            related_classes.add(c_idx)
            valid_edges_tc.append((tm_idx, c_idx))
            
        mask_g = (edge_tg[0] == tm_idx)
        g_indices = edge_tg[1][mask_g].tolist()
        for g_idx in g_indices:
            related_groups.add(g_idx)
            valid_edges_tg.append((tm_idx, g_idx))

    # 그래프 생성
    G = nx.Graph()

    # 노드 추가
    G.add_node(target_name, type='brand', size=2000)
    
    for tm_idx in my_tm_indices:
        raw_name = tm_names[tm_idx]
        short_name = raw_name.split('_')[0] 
        if len(short_name) > 10: short_name = short_name[:10] + ".."
        
        node_id = f"TM:{tm_idx}"
        G.add_node(node_id, label=short_name, type='trademark', size=800)
        G.add_edge(target_name, node_id)

    for c_idx in related_classes:
        name = class_names[c_idx]
        node_id = f"Class:{name}"
        G.add_node(node_id, label=f"{name}류", type='class', size=1200)
        for tm, c in valid_edges_tc:
            if c == c_idx: G.add_edge(f"TM:{tm}", node_id)

    for g_idx in related_groups:
        name = group_names[g_idx]
        node_id = f"Group:{name}"
        G.add_node(node_id, label=name, type='group', size=1000)
        for tm, g_val in valid_edges_tg:
            if g_val == g_idx: G.add_edge(f"TM:{tm}", node_id)

    # 시각화 캔버스 설정
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.7, iterations=50)

    color_map = {'brand': '#FF6B6B', 'trademark': '#4ECDC4', 'class': '#FFE66D', 'group': '#1A535C'}
    
    for n_type, color in color_map.items():
        n_list = [n for n, d in G.nodes(data=True) if d['type'] == n_type]
        sizes = [G.nodes[n]['size'] for n in n_list]
        nx.draw_networkx_nodes(G, pos, nodelist=n_list, node_color=color, node_size=sizes, alpha=0.9)

    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray')

    # ---------------------------------------------------------
    # 🎯 [핵심 수정] 폰트 이름을 NetworkX에 직접 전달
    # ---------------------------------------------------------
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    labels[target_name] = target_name 
    
    # font_family 인자에 위에서 찾은 'CHOSEN_FONT'를 직접 꽂아줍니다.
    # 이렇게 하면 전역 설정이 무시되더라도 이 폰트를 강제로 사용합니다.
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family=CHOSEN_FONT)

    # 범례
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Brand (중심)', markerfacecolor='#FF6B6B', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Trademark (상표)', markerfacecolor='#4ECDC4', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Class (류)', markerfacecolor='#FFE66D', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Group (유사군)', markerfacecolor='#1A535C', markersize=12)
    ]
    plt.legend(handles=legend_elements, loc='upper right', prop={'size': 12, 'family': CHOSEN_FONT})

    plt.title(f"Brand Graph Visualization: {target_name}", fontsize=15, fontfamily=CHOSEN_FONT)
    plt.axis('off')
    
    safe_name = "".join([c if c.isalnum() else "_" for c in target_name])
    output_img = f"./outputs/graph/viz_{safe_name}.png"
    
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 완료! 이미지가 저장되었습니다: {output_img}")
    plt.show()

if __name__ == "__main__":
    init_font() # 폰트 찾기
    data, encoders = load_data()
    visualize_ego_graph(data, encoders)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ==========================================
# ⚙️ 설정 (Settings)
# ==========================================
PYG_GRAPH_PATH = "./outputs/graph/pet_graph_data.pt"   # 원본 PyG 데이터 경로
DGL_GRAPH_PATH = "./outputs/graph/pet_graph_dgl.bin"   # 변환된 DGL 데이터 저장 경로
MODEL_SAVE_PATH = "./outputs/graph/dgl_gnn_model.pth"  # 학습된 모델 저장 경로
EMBEDDING_SAVE_PATH = "./outputs/graph/dgl_node_embeddings.pt" # 임베딩 벡터 저장 경로

HIDDEN_DIMS = 64   # 임베딩 차원 크기
EPOCHS = 500        # 학습 반복 횟수
LR = 0.001         # 학습률

# ==========================================
# 🛠️ 데이터 로드 및 변환 유틸리티
# ==========================================
def convert_pyg_to_dgl(pyg_path):
    """PyG 데이터를 읽어 DGL 그래프로 변환합니다."""
    print("🔄 PyG 데이터 로드 및 DGL 변환 시작 (최초 1회 실행)...")
    try:
        pyg_data = torch.load(pyg_path, weights_only=False)
    except TypeError:
        pyg_data = torch.load(pyg_path)
    
    data_dict = {}
    num_nodes_dict = {ntype: pyg_data[ntype].num_nodes for ntype in pyg_data.node_types}

    for edge_type in pyg_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = pyg_data[edge_type].edge_index
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        data_dict[(src_type, rel, dst_type)] = (src, dst)

    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    return g

def get_dgl_graph(force_reload=False):
    if os.path.exists(DGL_GRAPH_PATH) and not force_reload:
        print(f"✅ 캐시된 DGL 그래프를 로드합니다: {DGL_GRAPH_PATH}")
        g_list, _ = dgl.load_graphs(DGL_GRAPH_PATH)
        return g_list[0]
    else:
        if not os.path.exists(PYG_GRAPH_PATH):
            raise FileNotFoundError(f"❌ 원본 데이터가 없습니다: {PYG_GRAPH_PATH}")
        g = convert_pyg_to_dgl(PYG_GRAPH_PATH)
        print(f"💾 DGL 그래프를 캐싱합니다: {DGL_GRAPH_PATH}")
        os.makedirs(os.path.dirname(DGL_GRAPH_PATH), exist_ok=True)
        dgl.save_graphs(DGL_GRAPH_PATH, [g])
        return g

# ==========================================
# 🧠 GNN 모델 정의 (Hetero GraphSAGE)
# ==========================================
class HeteroSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats, out_feats):
        super().__init__()
        self.node_embeddings = nn.ModuleDict()
        for ntype in g.ntypes:
            self.node_embeddings[ntype] = nn.Embedding(g.num_nodes(ntype), in_feats)
        
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(in_feats, h_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum')
        
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(h_feats, out_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum')

    def forward(self, g, x_dict=None):
        # 1. 초기 임베딩 로드
        if x_dict is None:
            x_dict = {ntype: emb.weight for ntype, emb in self.node_embeddings.items()}
        
        # 2. 첫 번째 레이어 통과
        h1 = self.conv1(g, x_dict)
        
        # [핵심 수정] 사라진 노드 복구 (Residual Connection)
        # 회사 노드처럼 들어오는 엣지가 없는 경우 h1에 포함되지 않으므로, 원래 값을 넣어줍니다.
        for ntype in x_dict:
            if ntype not in h1:
                h1[ntype] = x_dict[ntype]
        
        h1 = {k: F.leaky_relu(v) for k, v in h1.items()}
        
        # 3. 두 번째 레이어 통과
        h2 = self.conv2(g, h1)
        
        # [핵심 수정] 2차 복구
        for ntype in h1:
            if ntype not in h2:
                h2[ntype] = h1[ntype]
                
        return h2

class ScorePredictor(nn.Module):
    """특정 타겟 엣지에 대해서만 점수 계산"""
    def forward(self, edge_subgraph, x, target_etype):
        with edge_subgraph.local_scope():
            src_type, _, dst_type = target_etype
            
            # 필요한 노드 타입 데이터 할당 (안전장치 추가)
            if src_type in edge_subgraph.ntypes:
                edge_subgraph.nodes[src_type].data['x'] = x[src_type]
            if dst_type in edge_subgraph.ntypes:
                edge_subgraph.nodes[dst_type].data['x'] = x[dst_type]

            # 타겟 엣지에 대해서만 연산
            edge_subgraph.apply_edges(fn.u_dot_v('x', 'x', 'score'), etype=target_etype)
            return edge_subgraph.edges[target_etype].data['score']

# ==========================================
# 🚀 메인 실행부
# ==========================================
if __name__ == "__main__":
    g = get_dgl_graph()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 학습 장치: {device}")
    
    g = g.to(device)

    target_etype = ('company', 'files', 'trademark')
    if target_etype not in g.canonical_etypes:
        # 데이터셋마다 엣지 이름이 다를 수 있어 유연하게 처리
        target_etype = g.canonical_etypes[0]
    
    print(f"🎯 학습 타겟 엣지: {target_etype}")

    model = HeteroSAGE(g, HIDDEN_DIMS, HIDDEN_DIMS, HIDDEN_DIMS).to(device)
    pred = ScorePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n🚀 DGL GNN 학습 시작...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # --- Negative Sampling ---
        src_node_count = g.num_nodes(target_etype[0])
        dst_node_count = g.num_nodes(target_etype[2])
        num_edges = g.num_edges(target_etype)
        
        neg_src = torch.randint(0, src_node_count, (num_edges,), device=device)
        neg_dst = torch.randint(0, dst_node_count, (num_edges,), device=device)
        
        neg_g = dgl.heterograph(
            {target_etype: (neg_src, neg_dst)},
            num_nodes_dict={nt: g.num_nodes(nt) for nt in g.ntypes}
        ).to(device)

        # --- Forward Pass ---
        h = model(g)
        
        # 수정된 ScorePredictor 호출 (인자 3개)
        pos_score = pred(g, h, target_etype)
        neg_score = pred(neg_g, h, target_etype)
        
        # --- Loss ---
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        # --- Backward ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- AUC 평가 ---
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                # AUC 계산 시 CPU로 이동
                auc = roc_auc_score(labels.cpu().numpy(), scores.sigmoid().cpu().numpy())
                print(f"Epoch: {epoch:03d}/{EPOCHS}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")

    print("\n💾 결과 저장 중...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    model.eval()
    with torch.no_grad():
        final_h = model(g)
        final_h_cpu = {k: v.cpu() for k, v in final_h.items()}
        torch.save(final_h_cpu, EMBEDDING_SAVE_PATH)
        
    print(f"✅ 학습 완료!")
    print(f" - 모델: {MODEL_SAVE_PATH}")
    print(f" - 임베딩: {EMBEDDING_SAVE_PATH}")
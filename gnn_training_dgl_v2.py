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
# ⚙️ 설정 (Settings) - 업그레이드
# ==========================================
PYG_GRAPH_PATH = "./outputs/graph/pet_graph_data.pt"
DGL_GRAPH_PATH = "./outputs/graph/pet_graph_dgl.bin"
MODEL_SAVE_PATH = "./outputs/graph/dgl_gnn_model_v2.pth"  # v2로 변경
EMBEDDING_SAVE_PATH = "./outputs/graph/dgl_node_embeddings_v2.pt" # v2로 변경

HIDDEN_DIMS = 128  # 🚀 [UP] 64 -> 128 (표현력 증가)
EPOCHS = 300       # MLP를 쓰면 수렴이 빨라질 수 있어 300으로 조정
LR = 0.0005        # 🚀 [DOWN] 정밀한 학습을 위해 학습률 약간 감소

# ==========================================
# 🛠️ 데이터 로드 유틸리티 (기존과 동일)
# ==========================================
def convert_pyg_to_dgl(pyg_path):
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
# 🧠 GNN 모델 정의 (3-Layer HeteroSAGE)
# ==========================================
class HeteroSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats, out_feats):
        super().__init__()
        self.node_embeddings = nn.ModuleDict()
        for ntype in g.ntypes:
            self.node_embeddings[ntype] = nn.Embedding(g.num_nodes(ntype), in_feats)
        
        # 🚀 [UP] 3층 구조로 변경 (Deep GNN)
        self.layers = nn.ModuleList()
        
        # Layer 1
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(in_feats, h_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum'))
        
        # Layer 2
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(h_feats, h_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum'))
        
        # Layer 3 (Output)
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(h_feats, out_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum'))

        self.dropout = nn.Dropout(0.5) # 과적합 방지

    def forward(self, g, x_dict=None):
        if x_dict is None:
            x_dict = {ntype: emb.weight for ntype, emb in self.node_embeddings.items()}
        
        h = x_dict
        
        # 레이어 반복 통과 (Residual Connection 포함)
        for i, layer in enumerate(self.layers):
            h_new = layer(g, h)
            
            # 노드 소실 방지 (Residual)
            for ntype in h:
                if ntype not in h_new:
                    h_new[ntype] = h[ntype] # 이전 값 유지
                else:
                    # 차원이 같을 때만 Residual 더하기 (Skip Connection)
                    if h[ntype].shape == h_new[ntype].shape:
                         h_new[ntype] = h_new[ntype] + h[ntype]
            
            h = h_new
            
            # 마지막 레이어가 아니면 Activation & Dropout 적용
            if i < len(self.layers) - 1:
                h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
                
        return h

# ==========================================
# 🧠 MLP Predictor (핵심 업그레이드)
# ==========================================
class MLPPredictor(nn.Module):
    """
    단순 내적(Dot Product) 대신 신경망을 사용하여 연결 확률 예측
    (h_u, h_v) -> Linear -> ReLU -> Linear -> Score
    """
    def __init__(self, h_feats):
        super().__init__()
        # 입력: 소스 노드 벡터 + 타겟 노드 벡터 (concat)
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        # 소스(src)와 타겟(dst) 벡터를 이어 붙임 (Concatenate)
        h = torch.cat([edges.src['x'], edges.dst['x']], dim=1)
        # MLP 통과
        score = self.W2(F.relu(self.W1(h))).squeeze(1)
        return {'score': score}

    def forward(self, edge_subgraph, x, target_etype):
        with edge_subgraph.local_scope():
            src_type, _, dst_type = target_etype
            
            # 노드 데이터 할당
            if src_type in edge_subgraph.ntypes:
                edge_subgraph.nodes[src_type].data['x'] = x[src_type]
            if dst_type in edge_subgraph.ntypes:
                edge_subgraph.nodes[dst_type].data['x'] = x[dst_type]

            # 타겟 엣지에 대해 apply_edges 함수 실행 (MLP 계산)
            edge_subgraph.apply_edges(self.apply_edges, etype=target_etype)
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
    
    # 엣지 확인
    if target_etype not in g.canonical_etypes:
        target_etype = g.canonical_etypes[0]
    
    print(f"🎯 학습 타겟: {target_etype}")
    print(f"🧠 모델 구성: 3-Layer HeteroSAGE + MLP Predictor (Hidden: {HIDDEN_DIMS})")

    # 모델 초기화 (v2)
    model = HeteroSAGE(g, HIDDEN_DIMS, HIDDEN_DIMS, HIDDEN_DIMS).to(device)
    pred = MLPPredictor(HIDDEN_DIMS).to(device) # MLP Predictor 사용
    
    # 두 모델의 파라미터를 모두 학습해야 함
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pred.parameters()), 
        lr=LR
    )

    print("\n🚀 V2 모델 학습 시작...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pred.train() # Predictor도 train 모드
        
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
        
        pos_score = pred(neg_g, h, target_etype) # neg_g 구조 재활용 (변수명 주의: 아래에서 g 사용)
        
        # DGL 버그 방지를 위해 정확한 그래프 객체 전달
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
        
        # --- AUC ---
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                auc = roc_auc_score(labels.cpu().numpy(), scores.sigmoid().cpu().numpy())
                print(f"Epoch: {epoch:03d}/{EPOCHS}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")

    print("\n💾 V2 결과 저장 중...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    model.eval()
    with torch.no_grad():
        final_h = model(g)
        final_h_cpu = {k: v.cpu() for k, v in final_h.items()}
        torch.save(final_h_cpu, EMBEDDING_SAVE_PATH)
        
    print(f"✅ V2 학습 완료!\n - 모델: {MODEL_SAVE_PATH}\n - 임베딩: {EMBEDDING_SAVE_PATH}")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from sklearn.metrics import roc_auc_score
import numpy as np

# ==========================================
# ⚙️ 설정
# ==========================================
PYG_GRAPH_PATH = "./outputs/graph/graph_data.pt"
MODEL_SAVE_PATH = "./outputs/graph/dgl_gnn_model_v3.pth"
EMBEDDING_SAVE_PATH = "./outputs/graph/dgl_node_embeddings_v3.pt"

HIDDEN_DIMS = 64
EPOCHS = 100
LR = 0.005

# ==========================================
# 🛠️ 그래프 로드 및 구조 변경 (Raw Data Direct Processing)
# ==========================================
def load_and_modify_graph():
    print("🔄 [Step 1] 원본 데이터 로드 및 지름길 계산 시작...")
    
    if not os.path.exists(PYG_GRAPH_PATH):
        raise FileNotFoundError(f"❌ 원본 데이터가 없습니다: {PYG_GRAPH_PATH}")
    
    # 1. PyG 데이터 로드 (안전하게 CPU로 로드)
    try:
        data = torch.load(PYG_GRAPH_PATH, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(PYG_GRAPH_PATH, map_location='cpu')

    # 2. 텐서 추출 (GPU가 아닌 CPU에서 안전하게 연산)
    print("   ↳ 엣지 데이터 추출 중 (CPU 안전 모드)...")
    
    # Company -> Trademark
    edge_ct = data['company', 'files', 'trademark'].edge_index
    src_ct = edge_ct[0] # Company ID
    dst_ct = edge_ct[1] # Trademark ID
    
    # Trademark -> Class
    edge_tc = data['trademark', 'belongs_to', 'class'].edge_index
    src_tc = edge_tc[0] # Trademark ID
    dst_tc = edge_tc[1] # Class ID

    # 노드 개수
    n_comp = data['company'].num_nodes
    n_tm = data['trademark'].num_nodes
    n_class = data['class'].num_nodes
    
    print(f"   ↳ 데이터 확인: Comp({n_comp}), TM({n_tm}), Class({n_class})")
    
    # -------------------------------------------------------
    # ⚡ [Shortcut] PyTorch CPU Sparse Matrix Multiplication
    # -------------------------------------------------------
    print("   ↳ 지름길 연산 수행 중 (Sparse MM)...")
    
    # 행렬 A (Comp x TM)
    indices_ct = torch.stack([src_ct, dst_ct])
    values_ct = torch.ones(len(src_ct))
    adj_ct = torch.sparse_coo_tensor(indices_ct, values_ct, (n_comp, n_tm))
    
    # 행렬 B (TM x Class)
    indices_tc = torch.stack([src_tc, dst_tc])
    values_tc = torch.ones(len(src_tc))
    adj_tc = torch.sparse_coo_tensor(indices_tc, values_tc, (n_tm, n_class))
    
    # 행렬 곱: (Comp x Class)
    # CPU Sparse MM은 매우 안정적입니다.
    adj_cc = torch.sparse.mm(adj_ct, adj_tc).coalesce()
    
    # 결과 추출
    indices_cc = adj_cc.indices()
    new_src = indices_cc[0]
    new_dst = indices_cc[1]
    
    count = len(new_src)
    print(f"   ✨ [성공] 지름길 생성 완료: {count:,}개의 (Company->Class) 직접 연결 발견!")
    
    if count < 100:
        raise ValueError("❌ 치명적 오류: 연결된 데이터가 거의 없습니다. 데이터 생성 과정을 점검하세요.")

    # -------------------------------------------------------
    # 🔄 [Step 2] DGL 그래프 생성
    # -------------------------------------------------------
    print("🔄 [Step 2] 학습용 DGL 그래프 생성 중...")
    
    data_dict = {}
    
    # 기존 엣지 옮기기
    # PyG Edge Index -> DGL (src, dst) Tuple
    if ('company', 'files', 'trademark') in data.edge_types:
        idx = data['company', 'files', 'trademark'].edge_index
        data_dict[('company', 'files', 'trademark')] = (idx[0].numpy(), idx[1].numpy())
        
    if ('trademark', 'belongs_to', 'class') in data.edge_types:
        idx = data['trademark', 'belongs_to', 'class'].edge_index
        data_dict[('trademark', 'belongs_to', 'class')] = (idx[0].numpy(), idx[1].numpy())

    # 새로운 지름길 엣지 추가
    data_dict[('company', 'interested_in', 'class')] = (new_src.numpy(), new_dst.numpy())
    
    # 노드 개수 명시
    num_nodes_dict = {
        'company': n_comp,
        'trademark': n_tm,
        'class': n_class
    }
    
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    return g

# ==========================================
# 🧠 모델 정의
# ==========================================
class SimpleHeteroSAGE(nn.Module):
    def __init__(self, g, in_feats, h_feats):
        super().__init__()
        self.node_embeddings = nn.ModuleDict()
        for ntype in g.ntypes:
            self.node_embeddings[ntype] = nn.Embedding(g.num_nodes(ntype), in_feats)
        
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(in_feats, h_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum')
        
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.SAGEConv(h_feats, h_feats, 'mean')
            for etype in g.etypes
        }, aggregate='sum')

    def forward(self, g, x_dict=None):
        if x_dict is None:
            x_dict = {ntype: emb.weight for ntype, emb in self.node_embeddings.items()}
        
        # Layer 1
        h1 = self.conv1(g, x_dict)
        # Residual Connection
        for ntype in x_dict:
            if ntype not in h1: h1[ntype] = x_dict[ntype]
        h1 = {k: F.leaky_relu(v) for k, v in h1.items()}
        
        # Layer 2
        h2 = self.conv2(g, h1)
        # Residual Connection 2
        for ntype in h1:
            if ntype not in h2: h2[ntype] = h1[ntype]
                
        return h2

class LinkPredictor(nn.Module):
    def forward(self, edge_subgraph, x, target_etype):
        with edge_subgraph.local_scope():
            src_type, _, dst_type = target_etype
            if src_type in edge_subgraph.ntypes:
                edge_subgraph.nodes[src_type].data['x'] = x[src_type]
            if dst_type in edge_subgraph.ntypes:
                edge_subgraph.nodes[dst_type].data['x'] = x[dst_type]

            edge_subgraph.apply_edges(fn.u_dot_v('x', 'x', 'score'), etype=target_etype)
            return edge_subgraph.edges[target_etype].data['score']

# ==========================================
# 🚀 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1. 그래프 생성 (캐시 없이 직접 생성)
    g = load_and_modify_graph()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 학습 장치: {device}")
    g = g.to(device)

    target_etype = ('company', 'interested_in', 'class')
    print(f"🎯 학습 타겟: {target_etype}")

    model = SimpleHeteroSAGE(g, HIDDEN_DIMS, HIDDEN_DIMS).to(device)
    pred = LinkPredictor().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=LR)

    print("\n🚀 V3 (Final Fix) 모델 학습 시작...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        src_n = g.num_nodes(target_etype[0])
        dst_n = g.num_nodes(target_etype[2])
        n_edges = g.num_edges(target_etype)
        
        if n_edges == 0:
            print("⚠️ 학습할 엣지가 없습니다!")
            break

        # Negative Sampling
        neg_src = torch.randint(0, src_n, (n_edges,), device=device)
        neg_dst = torch.randint(0, dst_n, (n_edges,), device=device)
        
        neg_g = dgl.heterograph(
            {target_etype: (neg_src, neg_dst)},
            num_nodes_dict={nt: g.num_nodes(nt) for nt in g.ntypes}
        ).to(device)

        h = model(g)
        
        pos_score = pred(g, h, target_etype)
        neg_score = pred(neg_g, h, target_etype)
        
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                auc = roc_auc_score(labels.cpu().numpy(), scores.sigmoid().cpu().numpy())
                print(f"Epoch: {epoch:03d}/{EPOCHS}, Loss: {loss.item():.4f}, AUC: {auc:.4f}")

    print("\n💾 V3 결과 저장 중...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    model.eval()
    with torch.no_grad():
        final_h = model(g)
        final_h_cpu = {k: v.cpu() for k, v in final_h.items()}
        torch.save(final_h_cpu, EMBEDDING_SAVE_PATH)
        
    print(f"✅ V3 학습 완료! 모델이 저장되었습니다.")
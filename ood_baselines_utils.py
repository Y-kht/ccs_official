import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from scipy.special import logsumexp
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================================
# Metrics Evaluation Functions
# ==========================================

def auroc(id_scores, ood_scores):
    """AUROC via Wilcoxon rank-sum (no sklearn needed)."""
    n0, n1 = len(id_scores), len(ood_scores)
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(n0), np.ones(n1)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    
    # Handle ties
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = ranks[order[i:j]].mean()
        ranks[order[i:j]] = avg_rank
        i = j
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)

def fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95):
    """FPR when TPR = tpr_target (OOD = positive class)."""
    threshold = np.percentile(ood_scores, 100 * (1 - tpr_target))
    return (id_scores >= threshold).mean()

def compute_ood_metrics(id_scores, ood_scores):
    """Return AUROC (%) and FPR@95 (%)."""
    return {
        'auroc': auroc(id_scores, ood_scores) * 100,
        'fpr95': fpr_at_tpr(id_scores, ood_scores) * 100,
    }

# ==========================================
# Helper: Safe FC Layer Management
# ==========================================
class FCManager:
    """Helper to remove and restore the FC layer seamlessly."""
    def __init__(self, model):
        self.model = model
        self.original_fc = None
        self.original_classifier = None
        self.layer_type = None

    def remove(self):
        if hasattr(self.model, 'fc'):
            self.original_fc = self.model.fc
            self.model.fc = nn.Identity()
            self.layer_type = 'fc'
        elif hasattr(self.model, 'classifier'):
            self.original_classifier = self.model.classifier
            self.model.classifier = nn.Identity()
            self.layer_type = 'classifier'

    def restore(self):
        if self.layer_type == 'fc':
            self.model.fc = self.original_fc
        elif self.layer_type == 'classifier':
            self.model.classifier = self.original_classifier

# ==========================================
# 1. COMBOOD
# ==========================================
def combood(model, id_dataset, ood_dataset, params):
    device = next(model.parameters()).device
    
    fc_manager = FCManager(model)
    fc_manager.remove()
    model.eval()
    
    activation_extrema = []
    def extrema_hook(module, input, output):
        x = input[0].view(input[0].size(0), -1) 
        mins = x.min(dim=1)[0]
        maxs = x.max(dim=1)[0]
        activation_extrema.append(torch.stack([mins, maxs], dim=1))
        
    handles = [m.register_forward_hook(extrema_hook) for m in model.modules() if isinstance(m, nn.ReLU)]
            
    def extract_features(dataloader):
        penultimate_feats, extrema_feats = [], []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                activation_extrema.clear() 
                
                out = model(x)
                norm_out = F.normalize(out, p=2, dim=1)
                penultimate_feats.append(norm_out.cpu().numpy())
                
                batch_extrema = torch.cat(activation_extrema, dim=1)
                extrema_feats.append(batch_extrema.cpu().numpy())
                
        return np.vstack(penultimate_feats), np.vstack(extrema_feats)

    train_loader = params.get('id_train_loader', id_dataset)
    train_penultimate, train_extrema = extract_features(train_loader)
    id_penultimate, id_extrema = extract_features(id_dataset)
    ood_penultimate, ood_extrema = extract_features(ood_dataset)
    
    for handle in handles: handle.remove()

    k = params.get('k', 50)
    C = params.get('C', 1.0)
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(train_penultimate)
    pt = PowerTransformer(method='yeo-johnson', standardize=True).fit(train_extrema)
    train_extrema_transformed = pt.transform(train_extrema)
    
    mu = np.mean(train_extrema_transformed, axis=0)
    cov = np.cov(train_extrema_transformed, rowvar=False)
    M_prime = cov + C * np.eye(cov.shape[0])
    M_prime_inv = np.linalg.inv(M_prime)
    
    d = train_extrema_transformed.shape[1]
    _, logdet = np.linalg.slogdet(M_prime)
    log_det_term = 0.5 * (d * np.log(2 * np.pi) + logdet)
    n_features = train_penultimate.shape[1]

    def compute_scores(penultimate_feats, extrema_feats):
        distances, _ = knn.kneighbors(penultimate_feats, n_neighbors=k)
        kd = np.maximum(distances[:, -1], 1e-8)
        kc = -np.sqrt(n_features) * np.log(kd)
        
        ext_transformed = pt.transform(extrema_feats)
        diff = ext_transformed - mu
        md_sq = np.sum(np.dot(diff, M_prime_inv) * diff, axis=1)
        mc = -0.5 * md_sq - log_det_term
        return -(kc + mc)

    id_scores = compute_scores(id_penultimate, id_extrema)
    ood_scores = compute_scores(ood_penultimate, ood_extrema)
    
    fc_manager.restore()
    return id_scores, ood_scores

# ==========================================
# 2. D-KNN (Dual-Space KNN)
# ==========================================
def d_knn(model, id_dataset, ood_dataset, params):
    device = next(model.parameters()).device
    
    fc_manager = FCManager(model)
    fc_manager.remove()
    model.eval()
    
    def extract_features(dataloader):
        feats = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                feats.append(model(x).cpu().numpy())
        return np.vstack(feats)

    train_loader = params.get('id_train_loader', id_dataset)
    X_train = extract_features(train_loader)
    X_id = extract_features(id_dataset)
    X_ood = extract_features(ood_dataset)

    def phi(x): return x / np.linalg.norm(x, axis=1, keepdims=True)

    k = params.get('k', 50)
    d = params.get('d', 0.95)
    alpha = params.get('alpha', 0.5)
    
    Z_train = phi(X_train)
    mu = np.mean(Z_train, axis=0, keepdims=True)
    Z_train_centered = Z_train - mu
    
    pca = PCA(n_components=d).fit(Z_train_centered)
    V = pca.components_.T
    P_prin = V @ V.T
    P_res = np.eye(Z_train.shape[1]) - P_prin
    
    def manifold_retract(Z_centered, P_matrix):
        return phi((Z_centered @ P_matrix) + mu)

    Z_train_prin = manifold_retract(Z_train_centered, P_prin)
    Z_train_res = manifold_retract(Z_train_centered, P_res)
    
    knn_p = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(Z_train_prin)
    s_p_train = knn_p.kneighbors(Z_train_prin)[0][:, k]
    mu_p, sigma_p = np.mean(s_p_train), np.std(s_p_train)
    
    knn_r = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(Z_train_res)
    s_r_train = knn_r.kneighbors(Z_train_res)[0][:, k]
    mu_r, sigma_r = np.mean(s_r_train), np.std(s_r_train)

    def compute_scores(X_test):
        Z_test_centered = phi(X_test) - mu
        Z_test_prin = manifold_retract(Z_test_centered, P_prin)
        Z_test_res = manifold_retract(Z_test_centered, P_res)
        
        s_p_test = knn_p.kneighbors(Z_test_prin, n_neighbors=k)[0][:, -1]
        s_r_test = knn_r.kneighbors(Z_test_res, n_neighbors=k)[0][:, -1]
        
        s_tilde_p = (s_p_test - mu_p) / (sigma_p + 1e-8)
        s_tilde_r = (s_r_test - mu_r) / (sigma_r + 1e-8)
        return alpha * s_tilde_p + (1 - alpha) * s_tilde_r

    id_scores = compute_scores(X_id)
    ood_scores = compute_scores(X_ood)
    
    fc_manager.restore()
    return id_scores, ood_scores

# ==========================================
# 3. CIDER
# ==========================================
def cider(model, id_dataset, ood_dataset, params):
    device = next(model.parameters()).device
    
    fc_manager = FCManager(model)
    fc_manager.remove()
    model.eval()
    
    train_loader = params['id_train_loader']
    
    def extract_features(dataloader, return_labels=False):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                feats.append(model(x.to(device)).cpu())
                if return_labels: labels.append(y.cpu())
        if return_labels: return torch.cat(feats), torch.cat(labels)
        return torch.cat(feats)

    train_feats, train_labels = extract_features(train_loader, return_labels=True)
    
    e_dim = train_feats.shape[1]
    d_dim = params.get('proj_dim', 128)
    num_classes = len(torch.unique(train_labels))
    
    projection_head = nn.Sequential(
        nn.Linear(e_dim, e_dim),
        nn.ReLU(),
        nn.Linear(e_dim, d_dim)
    ).to(device)
    
    prototypes = F.normalize(torch.randn(num_classes, d_dim), p=2, dim=1).to(device)
    
    epochs = params.get('epochs', 500)
    lr = params.get('lr', 0.5)
    tau = params.get('tau', 0.1)
    lambda_c = params.get('lambda_c', 2.0)
    alpha = params.get('alpha', 0.999)
    batch_size = params.get('batch_size', 512)
    
    optimizer = torch.optim.SGD(projection_head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    feat_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    projection_head.train()
    mask_eye = ~torch.eye(num_classes, dtype=torch.bool, device=device)
    
    for epoch in range(epochs):
        for batch_feats, batch_labels in feat_loader:
            batch_feats, batch_labels = batch_feats.to(device), batch_labels.to(device)
            z = F.normalize(projection_head(batch_feats), p=2, dim=1)
            
            with torch.no_grad():
                for c in range(num_classes):
                    class_mask = (batch_labels == c)
                    effective_alpha = alpha ** class_mask.sum().item() #we add this to account fo the batch normalisation in the next step without losing its efficiency.
                    if class_mask.sum() > 0:
                        class_z = z[class_mask].mean(dim=0)
                        prototypes[c] = F.normalize(effective_alpha * prototypes[c] + (1 - effective_alpha) * class_z, p=2, dim=0)
            
            logits = torch.matmul(z, prototypes.T) / tau
            l_comp = F.cross_entropy(logits, batch_labels)
            
            proto_sim_masked = (torch.matmul(prototypes, prototypes.T) / tau).masked_fill(~mask_eye, float('-inf'))
            l_dis = (torch.logsumexp(proto_sim_masked, dim=1) - math.log(num_classes - 1)).mean()
            
            loss = l_dis + lambda_c * l_comp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()

    projection_head.eval()
    def compute_scores(dataloader):
        scores = []
        with torch.no_grad():
            for x, _ in dataloader:
                z = F.normalize(projection_head(model(x.to(device))), p=2, dim=1)
                max_sim, _ = torch.max(torch.matmul(z, prototypes.T), dim=1)
                scores.extend((-max_sim).cpu().numpy())
        return np.array(scores)
    
    id_scores = compute_scores(id_dataset)
    ood_scores = compute_scores(ood_dataset)
    
    fc_manager.restore()
    return id_scores, ood_scores

# ==========================================
# 4. ViM (Virtual-logit Matching)
# ==========================================
def vim(model, id_dataset, ood_dataset, params):
    device = next(model.parameters()).device
    model.eval()
    
    # Needs FC layer, do not remove. Added 'linear' to the getattr chain.
    fc_layer = getattr(model, 'fc', getattr(model, 'classifier', getattr(model, 'linear', None)))
    if fc_layer is None:
        for m in model.modules():
            if isinstance(m, nn.Linear): fc_layer = m
                
    W = fc_layer.weight.detach().cpu().numpy()
    b = fc_layer.bias.detach().cpu().numpy()
    
    features_buffer = []
    def hook(module, input, output): features_buffer.append(input[0].detach().cpu().numpy())
    handle = fc_layer.register_forward_hook(hook)
    
    def extract_features_and_logits(dataloader):
        feats, logits = [], []
        with torch.no_grad():
            for x, _ in dataloader:
                features_buffer.clear()
                logits.append(model(x.to(device)).cpu().numpy())
                feats.append(features_buffer[0])
        return np.vstack(feats), np.vstack(logits)

    train_loader = params['id_train_loader']
    train_feats, train_logits = extract_features_and_logits(train_loader)
    id_feats, id_logits = extract_features_and_logits(id_dataset)
    ood_feats, ood_logits = extract_features_and_logits(ood_dataset)
    handle.remove()

    N_dim = train_feats.shape[1]
    if N_dim >= 1500:
        D = 1000
    elif N_dim <= 512:
        D = params.get('D', N_dim // 2)
    else:
        D = 512

    offset_o = -np.dot(np.linalg.pinv(W), b)
    X_train_centered = train_feats - offset_o
    
    eigvals, eigvecs = np.linalg.eigh(np.dot(X_train_centered.T, X_train_centered))
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    R = eigvecs[:, D:]
    
    train_res_norms = np.linalg.norm(np.dot(X_train_centered, R), axis=1)
    alpha = np.sum(np.max(train_logits, axis=1)) / (np.sum(train_res_norms) + 1e-8)
    
    def compute_vim_scores(feats, logits):
        res_norms = np.linalg.norm(np.dot(feats - offset_o, R), axis=1)
        return (alpha * res_norms) - logsumexp(logits, axis=1)

    return compute_vim_scores(id_feats, id_logits), compute_vim_scores(ood_feats, ood_logits)

# ==========================================
# 5. NAC (Neuron Activation Coverage)
# ==========================================
#Single FC_layer
def nac(model, id_dataset, ood_dataset, params):
    device = next(model.parameters()).device
    model.eval()
    
    # Needs FC layer, do not remove. Added 'linear' to the getattr chain.
    fc_layer = getattr(model, 'fc', getattr(model, 'classifier', getattr(model, 'linear', None)))
    if fc_layer is None:
        for m in model.modules():
            if isinstance(m, nn.Linear): fc_layer = m
                
    z_dict = {}
    def hook(module, input, output): z_dict['z'] = input[0]
    handle = fc_layer.register_forward_hook(hook)
    
    alpha = params.get('alpha', 100.0)
    
    def extract_neuron_states(dataloader):
        states = []
        for x, _ in dataloader:
            x = x.to(device).requires_grad_(True)
            logits = model(x)
            z = z_dict['z']
            
            # Efficient and stable gradient computation avoiding model.zero_grad() issues
            loss = -torch.mean(F.log_softmax(logits, dim=1), dim=1).sum()
            grad_z = torch.autograd.grad(loss, z, create_graph=False)[0]
            
            state = torch.sigmoid(alpha * z * grad_z)
            states.append(state.detach().cpu().numpy())
        return np.vstack(states)

    train_loader = params['id_train_loader']
    train_states = extract_neuron_states(train_loader)
    id_states = extract_neuron_states(id_dataset)
    ood_states = extract_neuron_states(ood_dataset)
    handle.remove()

    num_neurons = train_states.shape[1]
    num_bins = params.get('num_bins', 1000)
    r = params.get('r', 1.0)
    
    histograms, bin_edges_list = [], []
    for i in range(num_neurons):
        hist, edges = np.histogram(train_states[:, i], bins=num_bins, density=True)
        histograms.append(hist)
        bin_edges_list.append(edges)

    def compute_nac_scores(states):
        coverage_scores = np.zeros_like(states)
        for i in range(num_neurons):
            bin_idx = np.clip(np.digitize(states[:, i], bin_edges_list[i]) - 1, 0, num_bins - 1)
            density = histograms[i][bin_idx]
            
            # Zero coverage if state is completely outside train bounds
            out_of_bounds = (states[:, i] < bin_edges_list[i][0]) | (states[:, i] > bin_edges_list[i][-1])
            density[out_of_bounds] = 0.0
            coverage_scores[:, i] = np.minimum(density, r) / r
            
        return -np.mean(coverage_scores, axis=1)

    return compute_nac_scores(id_states), compute_nac_scores(ood_states)

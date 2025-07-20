from torch import nn
import torch
import torch.nn.functional as F
from optimal_control.control_cell import PDECell


class MSClassifyNet(nn.Module):
    def __init__(self, num_classes, wiring, in_features, out_features):
        super().__init__()

        # PDE cells for SC and FC
        self.wm_sc = PDECell(wiring, in_features)
        self.wm_fc = PDECell(wiring, in_features)

        # Coupled PDE sequence
        self.dual_PDE_sequence = AdvancedDualPDESequence(
            self.wm_sc,
            self.wm_fc,
            coupling_type='gated_asymmetric'
        )

        # Fusion module
        self.fusion_module = AdvancedFusionModule(
            feature_dim=in_features,
            fusion_type='cross_modal_attention'
        )

        # Adaptive weight learner
        self.adaptive_weights = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 3),
            nn.Softmax(dim=-1)
        )

        # Residual connection
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, sc, fc):
        feature_vec = x

        # PDE processing
        out_sc, out_fc, usc, ufc, B_sc, B_fc, lambdaB = self.dual_PDE_sequence(
            feature_vec.unsqueeze(0), feature_vec.unsqueeze(0), sc, fc
        )

        out_sc, out_fc = out_sc[-1], out_fc[-1]
        usc, ufc = usc[-1], ufc[-1]
        B_sc, B_fc = B_sc[-1], B_fc[-1]
        lambdaB = lambdaB[-1]

        # Fusion
        fused_output, *_ = self.fusion_module(out_sc, out_fc)

        # Weighted sum
        combined = torch.cat([out_sc, out_fc], dim=-1)
        weights = self.adaptive_weights(combined)
        final = weights[0] * out_sc + weights[1] * out_fc + weights[2] * fused_output

        # Residual connection
        output = final + torch.sigmoid(self.residual_alpha) * feature_vec

        return output, usc, ufc, B_sc, B_fc, lambdaB


class AdvancedDualPDESequence(nn.Module):
    def __init__(self, PDE_cell_sc, PDE_cell_fc, coupling_type='gated_asymmetric'):
        super().__init__()
        self.PDE_cell_sc = PDE_cell_sc
        self.PDE_cell_fc = PDE_cell_fc

        self.coupling_module = AdaptiveCouplingModule(
            PDE_cell_sc.state_size, coupling_type
        )

        self.neuron_weights = nn.Parameter(torch.ones(PDE_cell_sc.state_size) * 0.5)
        self.state_coupling = nn.Sequential(
            nn.Linear(PDE_cell_sc.state_size * 2, PDE_cell_sc.state_size),
            nn.Sigmoid()
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def compute_adaptive_coupling(self, state_sc, state_fc):
        combined = torch.cat([state_sc, state_fc], dim=-1)
        coupling = self.state_coupling(combined)
        weights = torch.sigmoid(self.neuron_weights)
        return torch.sigmoid(coupling * weights / self.temperature)

    def forward(self, x_sc, x_fc, sc_adj, fc_adj):
        device = x_sc.device
        seq_len = x_sc.size(0)

        hs_sc = torch.zeros((1, self.PDE_cell_sc.state_size), device=device)
        hs_fc = torch.zeros((1, self.PDE_cell_fc.state_size), device=device)

        out_sc = torch.empty(seq_len, self.PDE_cell_sc.output_size, device=device)
        out_fc = torch.empty(seq_len, self.PDE_cell_fc.output_size, device=device)
        B_sc = torch.empty(seq_len, self.PDE_cell_sc.state_size, self.PDE_cell_sc.state_size, device=device)
        B_fc = torch.empty(seq_len, self.PDE_cell_fc.state_size, self.PDE_cell_fc.state_size, device=device)
        uscs = torch.empty(seq_len, self.PDE_cell_sc.state_size, device=device)
        ufcs = torch.empty(seq_len, self.PDE_cell_fc.state_size, device=device)
        lambdas = torch.empty(seq_len, self.PDE_cell_sc.state_size, device=device)

        for t in range(seq_len):
            inp_sc, inp_fc = x_sc[t], x_fc[t]

            out_s, hs_raw_sc, B_s, us = self.PDE_cell_sc(inp_sc, sc_adj, hs_sc)
            out_f, hs_raw_fc, B_f, uf = self.PDE_cell_fc(inp_fc, fc_adj, hs_fc)

            lambdaB = self.compute_adaptive_coupling(hs_raw_sc, hs_raw_fc)

            hs_sc_c, hs_fc_c = self.coupling_module(hs_raw_sc, hs_raw_fc)
            hs_sc = (1 - lambdaB) * hs_raw_sc + lambdaB * hs_sc_c
            hs_fc = (1 - lambdaB) * hs_raw_fc + lambdaB * hs_fc_c

            out_sc[t], out_fc[t] = out_s, out_f
            B_sc[t], B_fc[t] = B_s, B_f
            uscs[t], ufcs[t] = us, uf
            lambdas[t] = lambdaB

        return out_sc, out_fc, uscs, ufcs, B_sc, B_fc, lambdas


class AdaptiveCouplingModule(nn.Module):
    def __init__(self, state_size, coupling_type='gated_asymmetric'):
        super().__init__()
        self.type = coupling_type

        if coupling_type == 'gated_asymmetric':
            self.sc_gate = nn.Sequential(
                nn.Linear(state_size * 2, state_size), nn.Sigmoid())
            self.fc_gate = nn.Sequential(
                nn.Linear(state_size * 2, state_size), nn.Sigmoid())
            self.tr_sc = nn.Linear(state_size, state_size)
            self.tr_fc = nn.Linear(state_size, state_size)

    def gated_asymmetric_coupling(self, state_sc, state_fc):
        g_sc = self.fc_gate(torch.cat([state_fc, state_sc], dim=-1))
        g_fc = self.sc_gate(torch.cat([state_sc, state_fc], dim=-1))
        return state_sc + g_fc * self.tr_fc(state_fc), state_fc + g_sc * self.tr_sc(state_sc)

    def forward(self, state_sc, state_fc):
        if self.type == 'gated_asymmetric':
            return self.gated_asymmetric_coupling(state_sc, state_fc)
        return state_sc, state_fc


class AdvancedFusionModule(nn.Module):
    def __init__(self, feature_dim, fusion_type='cross_modal_attention'):
        super().__init__()
        self.type = fusion_type

        if fusion_type == 'cross_modal_attention':
            self.attn = nn.MultiheadAttention(feature_dim, 8, batch_first=True)
            self.norm = nn.LayerNorm(feature_dim)
            self.fuse = nn.Sequential(
                nn.Linear(feature_dim * 3, feature_dim * 2), nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim), nn.ReLU())
        elif fusion_type == 'bilinear_pooling':
            self.bilinear = nn.Bilinear(feature_dim, feature_dim, feature_dim)
            self.post = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, feature_dim))

    def cross_modal_attention_fusion(self, sc, fc):
        sc_q, fc_q = sc.unsqueeze(0), fc.unsqueeze(0)
        sc_attn, _ = self.attn(sc_q, fc_q, fc_q)
        fc_attn, _ = self.attn(fc_q, sc_q, sc_q)
        sc_out = self.norm(sc + sc_attn.squeeze(0))
        fc_out = self.norm(fc + fc_attn.squeeze(0))
        fused = torch.cat([sc_out, fc_out, sc_out * fc_out], dim=-1)
        return self.fuse(fused), None, None

    def bilinear_pooling_fusion(self, sc, fc):
        return self.post(self.bilinear(sc, fc))

    def forward(self, sc, fc):
        if self.type == 'cross_modal_attention':
            return self.cross_modal_attention_fusion(sc, fc)
        elif self.type == 'bilinear_pooling':
            return self.bilinear_pooling_fusion(sc, fc), None, None
        return 0.5 * sc + 0.5 * fc, None, None

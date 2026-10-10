"""PDCAN architecture/scaling/loss from the supplied fixed-epoch notebook.
Original architecture is preserved; no scenario-selection or test tuning here.
"""
import os, random, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS=0
GRAD_CLIP_NORM=1.0
USE_AMP=False
SVAL_WEIGHTS={"mPSI":0.4,"HR":0.2,"Tskin":0.2,"Tcore":0.2}
USE_DATASET_TMAX_HRMAX=True

HP={'static_hidden': (64, 64), 'tcn_channels': (64, 64), 'tcn_kernel': 5, 'history_gru_hidden_dims': (128, 128), 'control_gru_hidden': 64, 'decoder_hidden_dims': (64, 128), 'dropout': 0.1, 'activation': 'silu', 'lr': 0.001, 'weight_decay': 1e-05, 'batch_size': 128, 'alpha_hr': 1.5, 'alpha_tskin': 1.0, 'alpha_tcore': 1.0, 'lambda_phys': 1.0, 'lambda_mpsi': 3.0, 'lambda_smooth': 0.0001, 'tf_start': 0.8, 'tf_end': 0.05}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@dataclass
class RawData:
    x_static: np.ndarray        # [N, 5]
    x_hist: np.ndarray          # [N, 30, 7]
    u_ctrl: np.ndarray          # [N, 15, 5]
    y_seq: np.ndarray           # [N, 15, 3], [HR,Tskin,Tcore]
    y_mpsi: np.ndarray          # [N, 1]
    init_phys: np.ndarray       # [N, 2], [HR_t,Tskin_t]
    meta: np.ndarray            # [N, 5], [T0,HR0,Tmax_col,HRmax_col,age]
    type_label: np.ndarray      # [N, 1], test intervention type: 1/2/3; train may be NaN
    sample_id: np.ndarray       # [N, 1], stable 0-based id after filtering, aligned with model arrays
    source_row: np.ndarray      # [N, 1], original CSV/Excel row number assuming header is row 1
    df: pd.DataFrame

def load_csv(path: str, name: str = "data") -> RawData:
    """
    读取训练集或测试集。

    前 340 列保持原始模型输入/标签格式：
    1-5 静态特征；6-215 历史 30×7；216-335 未来 15×8；
    336-339 meta；340 mPSI。

    如果文件存在第 341 列，则将其作为测试集 type 标签：
    type=1 仅休息干预；type=2 休息+环境条件变化；type=3 降低工作强度。
    该列不会进入模型训练，只用于测试集分组评价和绘图。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到{name}文件：{path}")
    df = pd.read_csv(path)
    # 保留原始行定位信息。source_row 对应 Excel/CSV 中的实际行号：表头为第1行，第一条数据为第2行。
    original_source_row = (np.arange(len(df), dtype=np.int64) + 2).reshape(-1, 1)
    if df.shape[1] < 340:
        raise ValueError(f"{name}至少应有 340 列，但当前只有 {df.shape[1]} 列。")

    arr = df.iloc[:, :340].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # 第 341 列为测试集干预类型，仅用于评估，不进入模型输入。
    if df.shape[1] >= 341:
        type_label = pd.to_numeric(df.iloc[:, 340], errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
        print(f"{name}: 检测到 type 列：{df.columns[340]}")
    else:
        type_label = np.full((len(df), 1), np.nan, dtype=np.float32)
        print(f"{name}: 未检测到第341列 type，分类型测试分析将跳过。")

    x_static = arr[:, 0:5]
    x_hist = arr[:, 5:215].reshape(-1, 30, 7)
    u_all = arr[:, 215:335].reshape(-1, 15, 8)

    u_ctrl = u_all[:, :, 0:5]
    y_seq = u_all[:, :, [7, 6, 5]]  # 输出顺序：HR、Tskin、Tcore
    init_phys = x_hist[:, -1, [6, 5]]  # HR_t、Tskin_t

    T0 = arr[:, 335:336]
    HR0 = arr[:, 336:337]
    Tmax_col = arr[:, 337:338]
    HRmax_col = arr[:, 338:339]
    age = x_static[:, 1:2]
    meta = np.concatenate([T0, HR0, Tmax_col, HRmax_col, age], axis=1)
    y_mpsi = arr[:, 339:340]

    valid = np.isfinite(y_seq).all(axis=(1, 2)) & np.isfinite(y_mpsi).all(axis=1)
    if valid.sum() < len(valid):
        print(f"{name}: 删除 {len(valid) - valid.sum()} 行关键标签缺失的数据。")
        x_static = x_static[valid]
        x_hist = x_hist[valid]
        u_ctrl = u_ctrl[valid]
        y_seq = y_seq[valid]
        y_mpsi = y_mpsi[valid]
        init_phys = init_phys[valid]
        meta = meta[valid]
        type_label = type_label[valid]
        original_source_row = original_source_row[valid]
        df = df.loc[valid].reset_index(drop=True)

    print(f"{name}读取完成：")
    print("  x_static:", x_static.shape)
    print("  x_hist  :", x_hist.shape)
    print("  u_ctrl  :", u_ctrl.shape)
    print("  y_seq   :", y_seq.shape, "[HR, Tskin, Tcore]")
    print("  y_mPSI  :", y_mpsi.shape)
    if np.isfinite(type_label).any():
        print("  type    :", pd.Series(type_label.reshape(-1)).value_counts(dropna=False).sort_index().to_dict())
    sample_id = np.arange(len(y_mpsi), dtype=np.int64).reshape(-1, 1)
    return RawData(x_static, x_hist, u_ctrl, y_seq, y_mpsi, init_phys, meta, type_label, sample_id, original_source_row, df)

class NanStandardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.scale_ = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.scale_ = np.nanstd(x, axis=0).astype(np.float32)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) * self.scale_ + self.mean_).astype(np.float32)

@dataclass
class Scalers:
    static: NanStandardizer
    hist: NanStandardizer
    ctrl: NanStandardizer
    y_phys: NanStandardizer
    y_mpsi: NanStandardizer

def fit_scalers(train_raw: RawData) -> Scalers:
    return Scalers(
        static=NanStandardizer().fit(train_raw.x_static),
        hist=NanStandardizer().fit(train_raw.x_hist.reshape(-1, 7)),
        ctrl=NanStandardizer().fit(train_raw.u_ctrl.reshape(-1, 5)),
        y_phys=NanStandardizer().fit(train_raw.y_seq.reshape(-1, 3)),
        y_mpsi=NanStandardizer().fit(train_raw.y_mpsi),
    )

def transform_raw(raw: RawData, scalers: Scalers) -> Dict[str, np.ndarray]:
    x_static = scalers.static.transform(raw.x_static)
    x_hist = scalers.hist.transform(raw.x_hist.reshape(-1, 7)).reshape(-1, 30, 7)
    u_ctrl = scalers.ctrl.transform(raw.u_ctrl.reshape(-1, 5)).reshape(-1, 15, 5)
    y_seq = scalers.y_phys.transform(raw.y_seq.reshape(-1, 3)).reshape(-1, 15, 3)
    y_mpsi = scalers.y_mpsi.transform(raw.y_mpsi)

    init_raw = raw.init_phys.copy()
    y_mean = scalers.y_phys.mean_
    y_std = scalers.y_phys.scale_
    init_phys = np.empty_like(init_raw, dtype=np.float32)
    init_phys[:, 0] = (init_raw[:, 0] - y_mean[0]) / y_std[0]
    init_phys[:, 1] = (init_raw[:, 1] - y_mean[1]) / y_std[1]
    init_phys = np.nan_to_num(init_phys, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return {
        "x_static": x_static,
        "x_hist": x_hist,
        "u_ctrl": u_ctrl,
        "y_seq": y_seq,
        "y_mpsi": y_mpsi,
        "init_phys": init_phys,
        "meta": np.nan_to_num(raw.meta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        # type_label、sample_id、source_row 只用于测试集分组分析和绘图，不参与模型计算。
        "type_label": raw.type_label.astype(np.float32),
        "sample_id": raw.sample_id.astype(np.float32),
        "source_row": raw.source_row.astype(np.float32),
    }

class PhysioDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_dict.items()}
        self.n = len(next(iter(self.data.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.items()}

def make_loader(data_dict: Dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        PhysioDataset(data_dict),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

def make_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int = None,
                 dropout: float = 0.1, activation: str = "gelu", layer_norm: bool = True):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        if out_dim is not None:
            dims.append(out_dim)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FormulaMPSIMixin:
    def formula_mpsi_norm(self, final_y_norm, meta):
        final_raw = final_y_norm * self.y_phys_std + self.y_phys_mean
        HR_pred = final_raw[:, 0:1]
        Tcore_pred = final_raw[:, 2:3]

        T0 = meta[:, 0:1]
        HR0 = meta[:, 1:2]
        Tmax_col = meta[:, 2:3]
        HRmax_col = meta[:, 3:4]
        age = meta[:, 4:5]

        Tmax_formula = torch.full_like(T0, 38.5)
        HRmax_formula = 180.0 - 0.65 * age

        if self.use_dataset_tmax_hrmax:
            valid_tmax = torch.isfinite(Tmax_col) & (Tmax_col > T0 + 1e-3)
            valid_hrmax = torch.isfinite(HRmax_col) & (HRmax_col > HR0 + 1e-3)
            Tmax = torch.where(valid_tmax, Tmax_col, Tmax_formula)
            HRmax = torch.where(valid_hrmax, HRmax_col, HRmax_formula)
        else:
            Tmax, HRmax = Tmax_formula, HRmax_formula

        denom_t = torch.clamp(Tmax - T0, min=1e-3)
        denom_hr = torch.clamp(HRmax - HR0, min=1e-3)
        mpsi_raw = 5.0 * (Tcore_pred - T0) / denom_t + 5.0 * (HR_pred - HR0) / denom_hr
        return (mpsi_raw - self.mpsi_mean) / self.mpsi_std

    def register_scaler_buffers(self, y_phys_mean, y_phys_std, mpsi_mean, mpsi_std):
        self.register_buffer("y_phys_mean", torch.tensor(y_phys_mean.reshape(1, 3), dtype=torch.float32))
        self.register_buffer("y_phys_std", torch.tensor(y_phys_std.reshape(1, 3), dtype=torch.float32))
        self.register_buffer("mpsi_mean", torch.tensor([[mpsi_mean]], dtype=torch.float32))
        self.register_buffer("mpsi_std", torch.tensor([[mpsi_std]], dtype=torch.float32))

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act = make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.resample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.dropout(y)
        return self.act(y + self.resample(x))

class TCN(nn.Module):
    def __init__(self, in_dim: int, channels: Tuple[int, ...], kernel_size: int = 3,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        layers = []
        prev = in_dim
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, kernel_size, dilation=2 ** i,
                                        dropout=dropout, activation=activation))
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.net(y)
        return y.transpose(1, 2)

class StackedVariableGRU(nn.Module):
    def __init__(self, input_size: int, hidden_dims: Tuple[int, ...], dropout: float = 0.1):
        super().__init__()
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        prev_dim = input_size
        for h in self.hidden_dims:
            self.layers.append(nn.GRU(input_size=prev_dim, hidden_size=h, num_layers=1, batch_first=True))
            prev_dim = h

    def forward(self, x):
        out = x
        last_h = None
        for i, gru in enumerate(self.layers):
            out, h = gru(out)
            last_h = h[-1]
            if i < len(self.layers) - 1:
                out = self.dropout(out)
        return out, last_h

class PDCANModel(nn.Module, FormulaMPSIMixin):
    def __init__(
        self,
        hp: Dict[str, Any],
        y_phys_mean: np.ndarray,
        y_phys_std: np.ndarray,
        mpsi_mean: float,
        mpsi_std: float,
        use_dataset_tmax_hrmax: bool = True,
        use_future_context: bool = True,
        use_autoregressive_feedback: bool = True,
        use_mpsi_residual: bool = True,
    ):
        super().__init__()
        self.hp = hp
        self.horizon = 15
        self.use_dataset_tmax_hrmax = use_dataset_tmax_hrmax
        self.use_future_context = use_future_context
        self.use_autoregressive_feedback = use_autoregressive_feedback
        self.use_mpsi_residual = use_mpsi_residual
        self.register_scaler_buffers(y_phys_mean, y_phys_std, mpsi_mean, mpsi_std)

        dropout = hp["dropout"]
        activation = hp.get("activation", "gelu")
        history_gru_hidden_dims = tuple(int(v) for v in hp["history_gru_hidden_dims"])
        self.decoder_hidden_dims = tuple(int(v) for v in hp["decoder_hidden_dims"])
        decoder_top_dim = self.decoder_hidden_dims[-1]
        fusion_hidden = decoder_top_dim

        static_hidden = tuple(hp["static_hidden"])
        self.static_encoder = MLP(
            5,
            static_hidden[:-1],
            out_dim=static_hidden[-1],
            dropout=dropout,
            activation=activation,
        )
        static_dim = static_hidden[-1]

        self.tcn = TCN(
            7,
            tuple(hp["tcn_channels"]),
            kernel_size=hp["tcn_kernel"],
            dropout=dropout,
            activation=activation,
        )
        self.hist_gru = StackedVariableGRU(self.tcn.out_dim, history_gru_hidden_dims, dropout=dropout)
        hist_dim = history_gru_hidden_dims[-1]

        self.ctrl_gru = nn.GRU(input_size=5, hidden_size=hp["control_gru_hidden"], num_layers=1, batch_first=True)
        ctrl_dim = hp["control_gru_hidden"]

        fusion_in = static_dim + hist_dim + ctrl_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            make_activation(activation),
        )
        self.decoder_init_proj = nn.Linear(fusion_hidden, sum(self.decoder_hidden_dims))

        decoder_input_dim = 5 + 3 + 1
        self.decoder_cells = nn.ModuleList()
        for i, h in enumerate(self.decoder_hidden_dims):
            in_dim = decoder_input_dim if i == 0 else self.decoder_hidden_dims[i - 1]
            self.decoder_cells.append(nn.GRUCell(input_size=in_dim, hidden_size=h))

        self.phys_head = nn.Sequential(
            nn.Linear(decoder_top_dim, decoder_top_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(decoder_top_dim, 3),
        )

        risk_hidden = max(decoder_top_dim // 2, 16)
        self.risk_residual = nn.Sequential(
            nn.Linear(decoder_top_dim + 3 + 1, risk_hidden),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(risk_hidden, 1),
        )

    def encode(self, x_static, x_hist, u_ctrl):
        e_static = self.static_encoder(x_static)
        h_tcn = self.tcn(x_hist)
        _, z_hist = self.hist_gru(h_tcn)
        _, h_ctrl = self.ctrl_gru(u_ctrl)
        c_fut = h_ctrl[-1]
        if not self.use_future_context:
            c_fut = torch.zeros_like(c_fut)
        return self.fusion(torch.cat([e_static, z_hist, c_fut], dim=-1))

    def forward(self, x_static, x_hist, u_ctrl, init_phys, meta, y_seq_true=None, teacher_forcing_ratio: float = 0.0):
        B = x_static.size(0)
        z0 = self.encode(x_static, x_hist, u_ctrl)
        h_flat = self.decoder_init_proj(z0)
        h_states = [h.contiguous() for h in torch.split(h_flat, self.decoder_hidden_dims, dim=-1)]

        prev_hr = init_phys[:, 0:1]
        prev_tskin = init_phys[:, 1:2]
        prev_tcore = torch.zeros_like(prev_hr)
        init_state = torch.cat([prev_hr, prev_tskin, prev_tcore], dim=-1)
        prev_state = init_state
        has_prev_core = torch.zeros(B, 1, device=x_static.device, dtype=x_static.dtype)

        preds = []
        h_top = h_states[-1]
        for k in range(self.horizon):
            dec_in = torch.cat([u_ctrl[:, k, :], prev_state, has_prev_core], dim=-1)
            layer_input = dec_in
            new_h = []
            for i, cell in enumerate(self.decoder_cells):
                h_i = cell(layer_input, h_states[i])
                new_h.append(h_i)
                layer_input = h_i
            h_states = new_h
            h_top = h_states[-1]
            y_k = self.phys_head(h_top)
            preds.append(y_k.unsqueeze(1))

            if k < self.horizon - 1:
                if self.use_autoregressive_feedback:
                    if self.training and y_seq_true is not None and teacher_forcing_ratio > 0:
                        use_teacher = (torch.rand(B, 1, device=x_static.device) < teacher_forcing_ratio).float()
                        prev_state = use_teacher * y_seq_true[:, k, :] + (1.0 - use_teacher) * y_k
                    else:
                        prev_state = y_k
                    has_prev_core = torch.ones_like(has_prev_core)
                else:
                    # Ablation: the decoder cannot feed the previous predicted physiology
                    # into the next step. It still sees minute-level future controls and
                    # its recurrent hidden state, but not the cascaded physiological output.
                    prev_state = init_state
                    has_prev_core = torch.zeros_like(has_prev_core)

        phys_seq = torch.cat(preds, dim=1)
        final_y = phys_seq[:, -1, :]
        mpsi_formula = self.formula_mpsi_norm(final_y, meta)
        if self.use_mpsi_residual:
            residual = self.risk_residual(torch.cat([h_top, final_y, mpsi_formula], dim=-1))
            mpsi = mpsi_formula + residual
        else:
            mpsi = mpsi_formula
        return {"phys_seq": phys_seq, "mpsi": mpsi, "mpsi_formula": mpsi_formula}

def to_device(batch: Dict[str, torch.Tensor], device=DEVICE):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], hp: Dict[str, Any]):
    pred_seq = outputs["phys_seq"]
    true_seq = batch["y_seq"]
    pred_mpsi = outputs["mpsi"]
    true_mpsi = batch["y_mpsi"]

    mse_seq = (pred_seq - true_seq) ** 2
    phys_loss = (
        hp["alpha_hr"] * mse_seq[:, :, 0].mean() +
        hp["alpha_tskin"] * mse_seq[:, :, 1].mean() +
        hp["alpha_tcore"] * mse_seq[:, :, 2].mean()
    )
    mpsi_loss = F.mse_loss(pred_mpsi, true_mpsi)

    if pred_seq.size(1) >= 3:
        d2 = pred_seq[:, 2:, :] - 2 * pred_seq[:, 1:-1, :] + pred_seq[:, :-2, :]
        smooth_loss = (d2 ** 2).mean()
    else:
        smooth_loss = torch.zeros((), device=pred_seq.device)

    total = hp["lambda_phys"] * phys_loss + hp["lambda_mpsi"] * mpsi_loss + hp["lambda_smooth"] * smooth_loss
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_phys": float(phys_loss.detach().cpu()),
        "loss_mpsi": float(mpsi_loss.detach().cpu()),
        "loss_smooth": float(smooth_loss.detach().cpu()),
    }

def inverse_phys(z: np.ndarray, scaler: NanStandardizer) -> np.ndarray:
    return scaler.inverse_transform(z.reshape(-1, 3)).reshape(z.shape)

def inverse_mpsi(z: np.ndarray, scaler: NanStandardizer) -> np.ndarray:
    return scaler.inverse_transform(z.reshape(-1, 1)).reshape(-1, 1)

def mae_rmse_r2_me(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    me = float(np.mean(err))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return mae, rmse, r2, me

def compute_raw_metrics(pred_seq_norm: np.ndarray, true_seq_norm: np.ndarray,
                        pred_mpsi_norm: np.ndarray, true_mpsi_norm: np.ndarray,
                        scalers: Scalers) -> Dict[str, float]:
    pred_seq = inverse_phys(pred_seq_norm, scalers.y_phys)
    true_seq = inverse_phys(true_seq_norm, scalers.y_phys)
    pred_mpsi = inverse_mpsi(pred_mpsi_norm, scalers.y_mpsi)
    true_mpsi = inverse_mpsi(true_mpsi_norm, scalers.y_mpsi)

    metrics = {}
    for j, name in enumerate(["HR", "Tskin", "Tcore"]):
        mae, rmse, r2, me = mae_rmse_r2_me(true_seq[:, :, j], pred_seq[:, :, j])
        metrics[f"MAE_{name}"] = mae
        metrics[f"RMSE_{name}"] = rmse
        metrics[f"R2_{name}"] = r2
        metrics[f"ME_{name}"] = me

    mae, rmse, r2, me = mae_rmse_r2_me(true_mpsi, pred_mpsi)
    metrics["MAE_mPSI"] = mae
    metrics["RMSE_mPSI"] = rmse
    metrics["R2_mPSI"] = r2
    metrics["ME_mPSI"] = me

    std_hr, std_tskin, std_tcore = scalers.y_phys.scale_.tolist()
    std_mpsi = float(scalers.y_mpsi.scale_[0])
    metrics["Sval"] = (
        SVAL_WEIGHTS["mPSI"] * metrics["MAE_mPSI"] / max(std_mpsi, 1e-8) +
        SVAL_WEIGHTS["HR"] * metrics["MAE_HR"] / max(std_hr, 1e-8) +
        SVAL_WEIGHTS["Tskin"] * metrics["MAE_Tskin"] / max(std_tskin, 1e-8) +
        SVAL_WEIGHTS["Tcore"] * metrics["MAE_Tcore"] / max(std_tcore, 1e-8)
    )
    return metrics

def compute_step_metrics(pred_seq_norm: np.ndarray, true_seq_norm: np.ndarray, scalers: Scalers) -> pd.DataFrame:
    pred_seq = inverse_phys(pred_seq_norm, scalers.y_phys)
    true_seq = inverse_phys(true_seq_norm, scalers.y_phys)
    rows = []
    for k in range(pred_seq.shape[1]):
        row = {"minute": k + 1}
        for j, name in enumerate(["HR", "Tskin", "Tcore"]):
            mae, rmse, r2, me = mae_rmse_r2_me(true_seq[:, k, j], pred_seq[:, k, j])
            row[f"MAE_{name}"] = mae
            row[f"RMSE_{name}"] = rmse
            row[f"R2_{name}"] = r2
            row[f"ME_{name}"] = me
        rows.append(row)
    return pd.DataFrame(rows)

def scheduled_teacher_ratio(epoch: int, max_epochs: int, hp: Dict[str, Any]) -> float:
    start = hp.get("tf_start", 1.0)
    end = hp.get("tf_end", 0.05)
    if max_epochs <= 1:
        return end
    frac = epoch / (max_epochs - 1)
    return float(start + (end - start) * frac)

def train_one_epoch(model, loader, optimizer, hp, epoch, total_epochs, scaler_amp=None):
    model.train()
    total_loss = 0.0
    n = 0
    tf_ratio = scheduled_teacher_ratio(epoch, total_epochs, hp)

    for batch in loader:
        batch = to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP and DEVICE.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(batch["x_static"], batch["x_hist"], batch["u_ctrl"],
                                batch["init_phys"], batch["meta"],
                                y_seq_true=batch["y_seq"], teacher_forcing_ratio=tf_ratio)
                loss, _ = compute_loss(outputs, batch, hp)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            outputs = model(batch["x_static"], batch["x_hist"], batch["u_ctrl"],
                            batch["init_phys"], batch["meta"],
                            y_seq_true=batch["y_seq"], teacher_forcing_ratio=tf_ratio)
            loss, _ = compute_loss(outputs, batch, hp)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        bs = batch["x_static"].size(0)
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
    return total_loss / max(n, 1), tf_ratio

def save_scalers(scalers: Scalers, path: str):
    state = {
        "static_mean": scalers.static.mean_.tolist(),
        "static_scale": scalers.static.scale_.tolist(),
        "hist_mean": scalers.hist.mean_.tolist(),
        "hist_scale": scalers.hist.scale_.tolist(),
        "ctrl_mean": scalers.ctrl.mean_.tolist(),
        "ctrl_scale": scalers.ctrl.scale_.tolist(),
        "y_phys_mean": scalers.y_phys.mean_.tolist(),
        "y_phys_scale": scalers.y_phys.scale_.tolist(),
        "y_mpsi_mean": scalers.y_mpsi.mean_.tolist(),
        "y_mpsi_scale": scalers.y_mpsi.scale_.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


SEED = 42
FIXED_EPOCHS = {"PDCAN": 50}
BEST_HYPERPARAMS = {"PDCAN": HP}

def build_model(model_name: str, hp: Dict[str, Any], scalers: Scalers):
    kwargs = dict(
        hp=hp,
        y_phys_mean=scalers.y_phys.mean_,
        y_phys_std=scalers.y_phys.scale_,
        mpsi_mean=float(scalers.y_mpsi.mean_[0]),
        mpsi_std=float(scalers.y_mpsi.scale_[0]),
        use_dataset_tmax_hrmax=USE_DATASET_TMAX_HRMAX,
    )
    name = model_name.lower()
    if name == "pdcan":
        return PDCANModel(
            **kwargs,
            use_future_context=True,
            use_autoregressive_feedback=False,
            use_mpsi_residual=True,
        )
    if name == "lstm":
        return RNNBaselineModel("lstm", **kwargs)
    if name == "gru":
        return RNNBaselineModel("gru", **kwargs)
    if name == "transformer":
        return TransformerAutoregressiveModel(**kwargs)
    raise ValueError(f"未知模型名称：{model_name}")

def load_checkpoint_model(scalers: Scalers, checkpoint_path: Path):
    model = build_model("PDCAN", BEST_HYPERPARAMS["PDCAN"], scalers).to(DEVICE)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"PDCAN checkpoint not found: {checkpoint_path}. "
            "Run with --stage train or --stage all first."
        )
    try:
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def predict_one_control_scenario(model, test_data, raw_u_ctrl, scalers, batch_size=128):
    """Run frozen PDCAN inference for one [N,15,5] raw control array."""
    u_ctrl_z = scalers.ctrl.transform(raw_u_ctrl.reshape(-1, 5)).reshape(raw_u_ctrl.shape)
    pred_seq_norm = []
    pred_mpsi_norm = []
    n = len(raw_u_ctrl)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x_static = torch.tensor(test_data["x_static"][start:stop], dtype=torch.float32, device=DEVICE)
        x_hist = torch.tensor(test_data["x_hist"][start:stop], dtype=torch.float32, device=DEVICE)
        u_ctrl = torch.tensor(u_ctrl_z[start:stop], dtype=torch.float32, device=DEVICE)
        init_phys = torch.tensor(test_data["init_phys"][start:stop], dtype=torch.float32, device=DEVICE)
        meta = torch.tensor(test_data["meta"][start:stop], dtype=torch.float32, device=DEVICE)
        outputs = model(
            x_static, x_hist, u_ctrl, init_phys, meta,
            y_seq_true=None, teacher_forcing_ratio=0.0,
        )
        pred_seq_norm.append(outputs["phys_seq"].detach().cpu().numpy())
        pred_mpsi_norm.append(outputs["mpsi"].detach().cpu().numpy())

    pred_seq_norm = np.concatenate(pred_seq_norm, axis=0)
    pred_mpsi_norm = np.concatenate(pred_mpsi_norm, axis=0)
    pred_seq = scalers.y_phys.inverse_transform(pred_seq_norm.reshape(-1, 3)).reshape(pred_seq_norm.shape)
    pred_mpsi = scalers.y_mpsi.inverse_transform(pred_mpsi_norm).reshape(-1)
    return pred_seq, pred_mpsi

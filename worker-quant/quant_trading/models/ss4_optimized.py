
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# 1) 工具函数
# =========================================================

def annualize_ret(daily_mean: float, periods: int = 252) -> float:
    return (1.0 + daily_mean) ** periods - 1.0

def annualize_vol(daily_std: float, periods: int = 252) -> float:
    return float(daily_std) * np.sqrt(periods)

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """投影到单纯形：w>=0, sum(w)=1"""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    if np.isclose(v.sum(), 1.0) and np.all(v >= 0):
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = 1.0
        return w
    rho = int(rho[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        w = np.zeros_like(v)
        w[int(np.argmax(v))] = 1.0
        return w
    return w / s

def shrink_cov(S: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """协方差收缩：S'=(1-d)S+d*diag(S)"""
    S = np.asarray(S, dtype=float)
    diag = np.diag(np.diag(S))
    return (1.0 - delta) * S + delta * diag

# =========================================================
# 2) 特征工程（Panel Ridge 输入）
# =========================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def slope_log_price(close: pd.Series, window: int = 60) -> pd.Series:
    """log(price) 的滚动线性回归斜率（简化，避免依赖 sklearn）"""
    x = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        if np.any(~np.isfinite(y)):
            return np.nan
        y = np.log(np.clip(y, 1e-12, None))
        x_ = x - x.mean()
        y_ = y - y.mean()
        denom = float((x_ * x_).sum())
        return float((x_ * y_).sum() / (denom + 1e-12))

    return close.rolling(window).apply(lambda a: _slope(np.asarray(a, dtype=float)), raw=False)

def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    for tkr in prices.columns:
        c = prices[tkr]
        df = pd.DataFrame(index=prices.index)
        # 动量
        df["ret1"] = c.pct_change()
        df["ret5"] = c.pct_change(5)
        df["ret20"] = c.pct_change(20)
        df["ret60"] = c.pct_change(60)
        # 波动
        df["vol20"] = df["ret1"].rolling(20).std()
        df["vol60"] = df["ret1"].rolling(60).std()
        # 均线偏离 + Z
        ma50 = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()
        df["ma_gap"] = (ma50 / (ma200 + 1e-12)) - 1.0
        df["z_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
        # 震荡/趋势
        df["rsi14"] = rsi(c, 14) / 100.0
        df["slope60"] = slope_log_price(c, 60)
        feats[tkr] = df

    out = pd.concat(feats, axis=1)  # columns: (ticker, feature)
    out = out.sort_index()
    return out

def make_target(prices: pd.DataFrame, H: int = 20) -> pd.DataFrame:
    """目标：预测 forward_sharpe ≈ fwd_return / vol20"""
    ret1 = prices.pct_change()
    fwd = prices.shift(-H) / prices - 1.0
    vol20 = ret1.rolling(20).std()
    return fwd / (vol20 + 1e-12)

# =========================================================
# 3) 模型 + 优化器
# =========================================================

class PanelRidge:
    """
    极简 Ridge：对特征做 z-score 标准化，并带截距项。
    """
    def __init__(self, alpha: float = 50.0):
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0) + 1e-12
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        n_feat = Xd.shape[1]
        I = np.eye(n_feat, dtype=float)
        I[0, 0] = 0.0  # 不惩罚截距
        A = Xd.T @ Xd + self.alpha * I
        b = Xd.T @ y
        self.beta_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        return Xd @ self.beta_

def solve_long_only_meanvar(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    lam: float = 5.0,
    gamma: float = 50.0,
    n_iter: int = 300
) -> np.ndarray:
    """
    long-only mean-variance with turnover smoothing:
      min_w  -mu'w + lam*w' Sigma w + gamma*||w-w_prev||^2
      s.t. w>=0, sum(w)=1
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    w = np.asarray(w_prev, dtype=float).copy()

    # Lipschitz 步长
    eig_max = float(np.linalg.eigvalsh(Sigma).max())
    L = 2.0 * lam * max(eig_max, 1e-12) + 2.0 * gamma
    step = 1.0 / L

    for _ in range(n_iter):
        grad = (-mu) + 2.0 * lam * (Sigma @ w) + 2.0 * gamma * (w - w_prev)
        w_new = project_to_simplex(w - step * grad)
        if float(np.linalg.norm(w_new - w)) < 1e-8:
            break
        w = w_new
    return w

# =========================================================
# 4) 回测引擎（含：熔断 + 权重漂移 + 成本）
# =========================================================

def _download_close_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        px = raw["Close"].copy()
    else:
        # 兼容某些情况下 yfinance 的返回结构
        px = raw.copy()

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.sort_index()
    return px

def backtest_multi_etf_circuit_breaker(
    tickers: list[str],
    benchmark_ticker: str = "1321.T",
    start: str = "2020-01-01",
    end: str | None = None,
    H: int = 20,
    train_window: int = 252,
    cov_lookback: int = 60,
    rebalance_every: int = 20,
    alpha: float = 10.0,
    lam: float = 2.0,
    gamma: float = 10.0,
    shrink_delta: float = 0.5,
    cost_bps: float = 3.0,
    ma_window: int = 60,
    min_valid_ratio: float = 0.98,
    min_cov_obs: int = 10,
):
    """
    返回：
      msg, w_df, px, trade_tickers, equity_series, risk_off_series, stats_df
    """
    print(f"1) 获取数据 (含基准 {benchmark_ticker}) ...")
    all_tickers = list(dict.fromkeys(list(tickers) + [benchmark_ticker]))

    try:
        px = _download_close_prices(all_tickers, start=start, end=end)
    except Exception as e:
        return f"数据下载失败: {e}", None, None, None, None, None, None

    # 清洗：允许少量缺口（避免 how='any' 把整列删掉）
    px = px.dropna(how="all")
    px = px.ffill()

    # 列有效率筛选
    valid_ratio = px.notna().mean(axis=0)
    keep_cols = valid_ratio[valid_ratio >= float(min_valid_ratio)].index.tolist()
    px = px[keep_cols].copy()

    trade_tickers = [t for t in tickers if t in px.columns]
    if benchmark_ticker not in px.columns:
        return f"缺失基准数据 {benchmark_ticker}", None, None, None, None, None, None
    if len(trade_tickers) == 0:
        return "交易标的均无可用数据（可能被清洗规则过滤）", None, None, None, None, None, None

    print(f"2) 构建特征/标签 (交易资产数: {len(trade_tickers)}) ...")
    trade_px = px[trade_tickers]
    feats = make_features(trade_px)
    y = make_target(trade_px, H=H)
    ret1 = trade_px.pct_change().fillna(0.0)

    # 熔断：基准 60 日线
    bench_px = px[benchmark_ticker].copy()
    bench_ma = bench_px.rolling(window=int(ma_window)).mean()

    # 对齐
    common_idx = feats.index.intersection(y.index).intersection(px.index)
    feats = feats.loc[common_idx]
    y = y.loc[common_idx]
    ret1 = ret1.loc[common_idx]
    bench_px = bench_px.loc[common_idx]
    bench_ma = bench_ma.loc[common_idx]
    ret_next = ret1.shift(-1)

    dates = common_idx
    n = len(trade_tickers)

    def panel_stack(date_idx: pd.Index):
        X_list, y_list = [], []
        for dt in date_idx:
            row = feats.loc[dt]
            yy_row = y.loc[dt]
            for tkr in trade_tickers:
                x = row[tkr].to_numpy(dtype=float)
                yy = float(yy_row.get(tkr, np.nan))
                if np.any(~np.isfinite(x)) or not np.isfinite(yy):
                    continue
                X_list.append(x)
                y_list.append(yy)
        if len(X_list) == 0:
            return None
        return np.vstack(X_list), np.asarray(y_list, dtype=float)

    model = PanelRidge(alpha=alpha)

    # 回测序列
    w = np.ones(n, dtype=float) / n  # 当前持仓权重（已包含漂移后的“真实仓位”）
    weights_exec = []               # 每日收盘执行后的目标权重（用于吃下一天收益）
    risk_off_flags = []
    turnover_list = []
    cost_list = []
    gross_ret_list = []
    net_ret_list = []

    # equity：用 next_dt 做索引，更贴近“净值在下一交易日收盘”
    equity_idx = [dates[train_window]]
    equity_vals = [1.0]

    print("3) 开始滚动回测 (Walk-Forward) ...")
    for i in range(train_window, len(dates) - 1):
        dt = dates[i]
        next_dt = dates[i + 1]

        # --- 熔断判定（若 MA 不足则默认 risk-off 更稳）
        px_b = float(bench_px.loc[dt])
        ma_b = float(bench_ma.loc[dt]) if np.isfinite(bench_ma.loc[dt]) else np.nan
        risk_off = (not np.isfinite(ma_b)) or (px_b < ma_b)

        # --- 生成今日收盘的目标权重（用于持有到 next_dt）
        w_target = w.copy()
        did_rebalance = False

        if risk_off:
            w_target = np.zeros(n, dtype=float)
        else:
            if (i - train_window) % int(rebalance_every) == 0:
                did_rebalance = True

                # 训练
                train_dates = dates[i - train_window: i]
                stacked = panel_stack(train_dates)
                if stacked is not None:
                    Xtr, ytr = stacked
                    model.fit(Xtr, ytr)

                # 预测（风险调整收益）
                row = feats.loc[dt]
                mu_ra = np.zeros(n, dtype=float)
                for k, tkr in enumerate(trade_tickers):
                    x = row[tkr].to_numpy(dtype=float).reshape(1, -1)
                    if np.any(~np.isfinite(x)):
                        mu_ra[k] = 0.0
                    else:
                        mu_ra[k] = float(model.predict(x)[0])

                # 波动率还原量级（明确对齐 tickers，避免顺序错位）
                vol20 = feats.loc[dt].xs("vol20", level=1).reindex(trade_tickers).to_numpy(dtype=float)
                vol20 = np.nan_to_num(vol20, nan=0.01, posinf=0.01, neginf=0.01)
                mu = mu_ra * vol20

                # 协方差（年化 + 收缩）
                rwin = ret1.iloc[max(i - cov_lookback, 0): i][trade_tickers].dropna()
                if len(rwin) >= int(min_cov_obs):
                    S = np.cov(rwin.to_numpy().T) * 252.0
                    Sigma = shrink_cov(S, delta=shrink_delta)
                else:
                    Sigma = np.eye(n, dtype=float)

                # 若从空仓恢复，给一个均仓起点
                w_prev_optim = w if w.sum() > 0.1 else (np.ones(n, dtype=float) / n)
                w_target = solve_long_only_meanvar(mu, Sigma, w_prev=w_prev_optim, lam=lam, gamma=gamma)

        # --- 交易成本：仅在权重“真的变化”时扣一次（调仓或熔断切换）
        trade_occurs = not np.allclose(w_target, w, atol=1e-12)
        turnover = float(np.abs(w_target - w).sum()) if trade_occurs else 0.0
        cost = turnover * (float(cost_bps) / 10000.0) if trade_occurs else 0.0

        # --- 下一日收益（现金为 0）
        r_next = ret_next.loc[dt, trade_tickers].to_numpy(dtype=float)
        r_next = np.nan_to_num(r_next, nan=0.0, posinf=0.0, neginf=0.0)

        gross_ret = 0.0 if risk_off else float(np.dot(w_target, r_next))
        net_ret = gross_ret - cost

        # --- 更新净值（next_dt）
        equity_vals.append(equity_vals[-1] * (1.0 + net_ret))
        equity_idx.append(next_dt)

        # --- 权重漂移：非调仓日不交易，但仓位会随收益自然漂移
        if risk_off:
            w = np.zeros(n, dtype=float)
        else:
            w = w_target * (1.0 + r_next)
            s = float(w.sum())
            if s > 1e-12:
                w = w / s
            else:
                w = np.ones(n, dtype=float) / n  # 极端情况兜底

        # --- 记录
        weights_exec.append(w_target.copy())
        risk_off_flags.append(bool(risk_off))
        turnover_list.append(turnover)
        cost_list.append(cost)
        gross_ret_list.append(gross_ret)
        net_ret_list.append(net_ret)

    w_df = pd.DataFrame(weights_exec, index=dates[train_window:len(dates)-1], columns=trade_tickers)
    equity = pd.Series(equity_vals, index=pd.Index(equity_idx, name="date"), name="equity")
    risk_off_series = pd.Series(risk_off_flags, index=w_df.index, name="risk_off")

    stats_df = pd.DataFrame(
        {
            "gross_ret": gross_ret_list,
            "net_ret": net_ret_list,
            "turnover": turnover_list,
            "cost": cost_list,
            "risk_off": risk_off_flags,
        },
        index=w_df.index,
    )

    return "Success", w_df, px, trade_tickers, equity, risk_off_series, stats_df


# =========================================================
# 5) 主程序
# =========================================================

if __name__ == "__main__":
    INITIAL_CAPITAL = 1_000_000  # JPY
    TICKERS = ["8306.T", "8316.T", "8411.T"]
    BENCHMARK = "1321.T"

    print("=" * 60)
    print("🚀 启动增强型量化策略 (修复版)")
    print(f"本金: {INITIAL_CAPITAL:,} JPY")
    print("=" * 60)

    # 运行回测
    res = backtest_multi_etf_circuit_breaker(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start="2020-01-01",
        H=20,
        rebalance_every=20,
        cost_bps=3.0,
        ma_window=60,
        min_valid_ratio=0.98,
    )
    
    msg, w_df, px, trade_tickers, equity, risk_off_s, stats = res

    if msg == "Success":
        # --- 1. 计算指标 ---
        final_equity = float(equity.iloc[-1]) * INITIAL_CAPITAL
        ret_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
        max_dd = max_drawdown(equity) * 100.0

        # --- 2. Plotly 绘图 (修复了导入和渲染逻辑) ---
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            subplot_titles=("Strategy Equity Curve (Relative)", "Risk-Off Status (1=Cash, 0=Invested)"),
            row_heights=[0.7, 0.3]
        )

        # 净值线
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, mode='lines', name='Equity', line=dict(color='#2ca02c')),
            row=1, col=1
        )

        # 熔断状态
        fig.add_trace(
            go.Scatter(x=risk_off_s.index, y=risk_off_s.astype(int), mode='lines', name='Risk-Off', fill='tozeroy', line=dict(color='red')),
            row=2, col=1
        )

        fig.update_layout(
            height=800, 
            title_text=f"Backtest Report | Return: {ret_pct:.2f}% | MaxDD: {max_dd:.2f}%",
            template="plotly_dark", # 换成深色模式，对眼睛更好
            showlegend=False
        )
        
        # 导出 HTML
        report_name = "strategy_report.html"
        fig.write_html(report_name)
        
        print("\n📊 最终战报:")
        print(f"最终资产: {int(final_equity):,} JPY")
        print(f"总收益率: {ret_pct:.2f}%")
        print(f"最大回撤: {max_dd:.2f}%")
        print(f"✅ 交互式报告已保存至: {report_name}")

        # --- 3. 明日建议 ---
        print("-" * 60)
        print(f"🔮 [明日实盘持仓建议] 日期: {w_df.index[-1].date()}")
        if bool(risk_off_s.iloc[-1]):
            print("⚠️ 熔断触发：空仓持有现金")
        else:
            latest_w = w_df.iloc[-1]
            latest_px = px[trade_tickers].iloc[-1]
            for tkr in trade_tickers:
                if latest_w[tkr] > 0.01:
                    amt = INITIAL_CAPITAL * latest_w[tkr]
                    shares = int(amt // latest_px[tkr])
                    print(f"  ● {tkr}: {latest_w[tkr]*100:.2f}% -> 约 {int(amt):,} JPY ({shares} 股)")

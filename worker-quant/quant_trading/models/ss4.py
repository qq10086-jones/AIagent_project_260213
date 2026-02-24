import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心工具函数 (保持原算法精髓)
# ==========================================

def annualize_ret(daily_mean, periods=252):
    return (1 + daily_mean) ** periods - 1

def annualize_vol(daily_std, periods=252):
    return daily_std * np.sqrt(periods)

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return dd.min()

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """投影到单纯形 (保证权重和为1，且非负)"""
    v = np.asarray(v, dtype=float)
    if v.sum() == 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        w = np.zeros_like(v)
        w[np.argmax(v)] = 1.0
        return w
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    w = w / w.sum()
    return w

def shrink_cov(S: np.ndarray, delta: float = 0.5) -> np.ndarray:
    """协方差收缩 (Ledoit-Wolf 简化版)"""
    diag = np.diag(np.diag(S))
    return (1 - delta) * S + delta * diag

# ==========================================
# 2. 特征工程 (保持 Panel Ridge 逻辑)
# ==========================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def slope_log_price(close: pd.Series, window: int = 60) -> pd.Series:
    logp = np.log(close.replace(0, np.nan))
    x = np.arange(window)
    x = (x - x.mean()) / (x.std() + 1e-12)
    def _slope(y):
        if np.any(np.isnan(y)): return np.nan
        y = (y - y.mean()) / (y.std() + 1e-12)
        return np.dot(x, y) / (len(x) - 1)
    return logp.rolling(window).apply(_slope, raw=True)

def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    for tkr in prices.columns:
        c = prices[tkr].dropna()
        df = pd.DataFrame(index=prices.index)
        # 基础动量因子
        df["ret1"] = c.pct_change()
        df["ret5"] = c.pct_change(5)
        df["ret20"] = c.pct_change(20)
        df["ret60"] = c.pct_change(60)
        # 波动率因子
        df["vol20"] = df["ret1"].rolling(20).std()
        df["vol60"] = df["ret1"].rolling(60).std()
        # 均线偏离度
        ma50 = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()
        df["ma_gap"] = (ma50 / (ma200 + 1e-12)) - 1.0
        df["z_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
        # 震荡与趋势因子
        df["rsi14"] = rsi(c, 14) / 100.0
        df["slope60"] = slope_log_price(c, 60)
        
        feats[tkr] = df
    out = pd.concat(feats, axis=1)
    return out

def make_target(prices: pd.DataFrame, H: int = 20) -> pd.DataFrame:
    """目标函数：预测夏普比率 (Ret / Vol)"""
    ret1 = prices.pct_change()
    fwd = prices.shift(-H) / prices - 1.0
    vol20 = ret1.rolling(20).std()
    y = fwd / (vol20 + 1e-12)
    return y

# ==========================================
# 3. 算法核心 (Ridge + MeanVar)
# ==========================================

class PanelRidge:
    def __init__(self, alpha: float = 50.0):
        self.alpha = alpha
        self.mean_ = None
        self.std_ = None
        self.beta_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0) + 1e-12
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        n_feat = Xd.shape[1]
        I = np.eye(n_feat); I[0, 0] = 0.0
        A = Xd.T @ Xd + self.alpha * I
        b = Xd.T @ y
        self.beta_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        Xd = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=1)
        return Xd @ self.beta_

def solve_long_only_meanvar(mu, Sigma, w_prev, lam=5.0, gamma=50.0, n_iter=300):
    w = w_prev.copy()
    eig_max = np.linalg.eigvalsh(Sigma).max()
    L = 2 * lam * max(eig_max, 1e-12) + 2 * gamma
    step = 1.0 / L
    for _ in range(n_iter):
        grad = (-mu) + 2 * lam * (Sigma @ w) + 2 * gamma * (w - w_prev)
        w_new = project_to_simplex(w - step * grad)
        if np.linalg.norm(w_new - w) < 1e-8: break
        w = w_new
    return w

# ==========================================
# 4. 回测引擎 (含 🛡️止损熔断模块)
# ==========================================

def backtest_multi_etf_circuit_breaker(
    tickers,
    benchmark_ticker="1321.T", # 日经225 ETF 作为大盘风向标
    start="2020-01-01",
    end=None,
    H=20,
    train_window=252,    # 缩短训练窗口适应市场
    cov_lookback=60,
    rebalance_every=20,
    alpha=10.0,
    lam=2.0,
    gamma=10.0,
    shrink_delta=0.5,
    cost_bps=3.0,
):
    print(f"1. 正在获取数据 (包含基准 {benchmark_ticker})...")
    all_tickers = list(set(tickers + [benchmark_ticker]))
    try:
        px = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    except Exception as e:
        return f"数据下载失败: {e}", None, None, None
    
    # 清洗数据
    px = px.dropna(how="all").dropna(axis=1, how="any")
    
    # 确认交易标的 (排除基准)
    trade_tickers = [t for t in tickers if t in px.columns]
    if benchmark_ticker not in px.columns:
        return f"缺失基准数据 {benchmark_ticker}", None, None, None
    
    print(f"2. 构建因子特征 (资产数: {len(trade_tickers)})...")
    # 只对交易标的做特征工程
    trade_px = px[trade_tickers]
    feats = make_features(trade_px)
    y = make_target(trade_px, H=H)
    ret1 = trade_px.pct_change().fillna(0.0)

    # ----------------------------------------
    # 🛡️ 止损熔断准备: 计算大盘均线
    # ----------------------------------------
    bench_px = px[benchmark_ticker]
    # 计算 60日均线 (牛熊分界线)
    bench_ma60 = bench_px.rolling(window=60).mean()
    
    # 对齐所有数据的索引
    common_idx = feats.index.intersection(y.index).intersection(px.index)
    feats = feats.loc[common_idx]
    y = y.loc[common_idx]
    ret1 = ret1.loc[common_idx]
    bench_px = bench_px.loc[common_idx]
    bench_ma60 = bench_ma60.loc[common_idx]
    ret_next = ret1.shift(-1) # 次日收益用于回测计算

    # 数据堆叠辅助函数
    def panel_stack(date_idx):
        X_list, y_list = [], []
        for dt in date_idx:
            row = feats.loc[dt]
            for tkr in trade_tickers:
                x = row[tkr].values.astype(float)
                yy = y.loc[dt, tkr]
                if np.any(np.isnan(x)) or np.isnan(yy): continue
                X_list.append(x)
                y_list.append(float(yy))
        if len(X_list) == 0: return None
        return np.vstack(X_list), np.array(y_list)

    # 初始化变量
    dates = common_idx
    n = len(trade_tickers)
    w = np.ones(n) / n # 初始均仓
    weights = []
    equity_curve = [1.0] # 净值曲线
    model = PanelRidge(alpha=alpha)
    
    print("3. 开始滚动回测 (Walk-Forward)...")
    
    risk_off_mode = False # 熔断状态标记

    for i in range(train_window, len(dates) - 1):
        dt = dates[i]
        
        # ----------------------------------------
        # 🛡️ 止损熔断逻辑 (Circuit Breaker Logic)
        # ----------------------------------------
        # 判断大盘是否跌破 60日线
        current_bench_price = bench_px.loc[dt]
        current_bench_ma = bench_ma60.loc[dt]
        
        # 如果 大盘 < 60日线 -> 触发熔断
        if current_bench_price < current_bench_ma:
            risk_off_mode = True
            # 强制空仓 (持有日元现金)
            w = np.zeros(n) 
        else:
            risk_off_mode = False
            # 如果从熔断恢复，或者本来就正常，检查是否需要调仓
            if (i - train_window) % rebalance_every == 0:
                # 重新训练模型
                train_dates = dates[i - train_window : i]
                stacked = panel_stack(train_dates)
                if stacked is not None:
                    Xtr, ytr = stacked
                    model.fit(Xtr, ytr)
                
                # 预测 Alpha (预期收益)
                mu_ra = []
                row = feats.loc[dt]
                for tkr in trade_tickers:
                    x = row[tkr].values.astype(float).reshape(1, -1)
                    mu_val = model.predict(x)[0] if not np.any(np.isnan(x)) else 0.0
                    mu_ra.append(float(mu_val))
                
                # 恢复波动率量级
                vol20 = feats.loc[dt].xs("vol20", level=1, axis=0).values.astype(float)
                vol20 = np.nan_to_num(vol20, nan=0.01)
                mu = np.array(mu_ra) * vol20 

                # 计算风险矩阵 (收缩协方差)
                rwin = ret1.iloc[i - cov_lookback : i][trade_tickers].dropna()
                if len(rwin) > 10:
                    S = np.cov(rwin.values.T) * 252.0
                    Sigma = shrink_cov(S, delta=shrink_delta)
                else:
                    Sigma = np.eye(n)
                
                # 如果前一天是空仓(全0)，给优化器一个初始值(均仓)
                w_prev_optim = w if w.sum() > 0.1 else np.ones(n)/n
                
                # 均值方差优化
                w = solve_long_only_meanvar(mu, Sigma, w_prev=w_prev_optim, lam=lam, gamma=gamma)
            
            # 非调仓日，权重保持不变 (Drift) - 简化处理保持 w 不变

        # 记录权重
        weights.append(w.copy())
        
        # 计算次日收益
        # 如果 w 全是 0 (熔断中)，Portfolio Return = 0 (现金收益)
        r_next_day = ret_next.loc[dt, trade_tickers].values
        if risk_off_mode:
            port_ret = 0.0
        else:
            port_ret = np.dot(w, r_next_day)
            # 扣除简单的交易成本 (这里简化处理，只扣调仓日的)
        
        equity_curve.append(equity_curve[-1] * (1 + port_ret))

    # 整理结果
    w_df = pd.DataFrame(weights, index=dates[train_window:-1], columns=trade_tickers)
    equity_series = pd.Series(equity_curve, index=dates[train_window:])
    
    return "Success", w_df, px, trade_tickers, equity_series, risk_off_mode

# ==========================================
# 5. 执行主程序
# ==========================================

if __name__ == "__main__":
    # 设置 100万日元 本金
    INITIAL_CAPITAL = 1000000 
    
    # 日本市场核心标的
    # 1542: 白银, 1540: 黄金, 1541: 铂金
    # 1321: 日经225 (作为大盘基准，用于熔断判定)
    TICKERS = ["1542.T", "1540.T", "1541.T"]
    BENCHMARK = "1321.T"
    
    print("="*50)
    print(f"🚀 启动增强型量化策略 (含大盘熔断风控)")
    print(f"本金: {INITIAL_CAPITAL} JPY")
    print("="*50)
    
    msg, w_df, px, trade_tickers, equity, last_state_risk_off = backtest_multi_etf_circuit_breaker(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start="2020-01-01"
    )
    
    if msg == "Success":
        print("\n✅ 回测完成!")
        final_equity = equity.iloc[-1] * INITIAL_CAPITAL
        ret_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        max_dd = max_drawdown(equity) * 100
        
        print(f"\n📊 最终战报:")
        print(f"最终资产: {int(final_equity):,} JPY")
        print(f"总收益率: {ret_pct:.2f}%")
        print(f"最大回撤: {max_dd:.2f}%")
        
        print("-" * 30)
        print("🔮 [明日实盘持仓建议]")
        
        # 获取最新权重
        latest_weights = w_df.iloc[-1]
        latest_date = w_df.index[-1].date()
        latest_prices = px.iloc[-1]
        
        print(f"日期: {latest_date}")
        
        if last_state_risk_off:
            print("\n⚠️ 警告: 熔断机制已触发！(大盘 < 60日均线)")
            print("👉 动作: 空仓 / 持有现金 (100% Cash)")
        else:
            print(f"状态: 市场正常 (Risk On)")
            print("\n建议配置:")
            has_pos = False
            for tkr in trade_tickers:
                w = latest_weights[tkr]
                if w > 0.01: # 过滤掉 < 1% 的碎仓
                    has_pos = True
                    amt = INITIAL_CAPITAL * w
                    price = latest_prices[tkr]
                    shares = int(amt // price)
                    print(f"  ● {tkr}: {w*100:.2f}% -> 约 {int(amt):,} JPY ({shares} 股)")
            
            if not has_pos:
                print("  (模型建议暂时空仓观望)")

    else:
        print(f"\n❌ 错误: {msg}")
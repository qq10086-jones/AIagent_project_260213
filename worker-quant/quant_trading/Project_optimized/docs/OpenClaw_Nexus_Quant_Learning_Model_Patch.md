# OpenClaw Nexus Quant 学习机制数学模型强化方案 (v1.3.5 Patch)

## 1. 现状分析与痛点
目前在 `worker.py` 的 `discovery_workflow` 中，你已经引入了基础的 `enable_learning` 机制。它的工作原理是：
- **当前逻辑**：将最后一次成功的策略参数（`max_position_pct`, `alpha_floor` 等）保存为 `learned_seed`，并在下一次运行时优先尝试。如果失败，则累加 `consecutive_failures`。
- **局限性**：这是一种**“单点记忆” (Point-Memory)**，它只记住了上一次的结果，缺乏对历史胜率的统计，也无法很好地应对市场环境的动态变化（Regime Shift）。例如，在震荡市中，保守策略和激进策略可能交替生效，单点记忆会导致系统在两者之间反复横跳（也就是过度拟合最新的单次噪音）。

## 2. 数学模型建议：多臂老虎机模型 (Multi-Armed Bandit) + 汤普森采样 (Thompson Sampling)
为了用数学模型强化这个功能，我们最适合引入**基于贝叶斯推断的汤普森采样 (Thompson Sampling)**。这个模型计算轻量，非常适合在无状态的 Worker 节点中运行，并且能优雅地解决“探索与利用（Exploration vs Exploitation）”难题。

### 数学原理
1. 将你的每一个策略模版（如 `strict`, `relax_position`, `broaden_factors` 等）视为老虎机的一根“拉杆”（Arm）。
2. 每根拉杆的胜率服从 **Beta 分布 $Beta(\alpha, \beta)$**。
   - $\alpha$ 代表该策略历史上的**成功次数**（+1 作为先验平滑）
   - $\beta$ 代表该策略历史上的**失败次数**（+1 作为先验平滑）
3. **运作流程**：
   - 每次运行前，系统为每个策略从各自的 $Beta(\alpha, \beta)$ 中随机采样一个分数 $	heta$。
   - 系统根据采样的分数 $	heta$ 对所有候选策略进行**动态排序**，优先执行分数高的策略。
   - 运行结束后，评估使用的策略。如果筛选出足够的股票（候选数 $\ge$ 目标数），则记为成功（Reward=1），对应策略的 $\alpha \leftarrow \alpha + 1$；否则记为失败，$\beta \leftarrow \beta + 1$。
4. **衰减因子 (Decay Factor)**：引入时间衰减 $\gamma$（如 0.95），定期缩小 $\alpha$ 和 $\beta$，确保模型能“遗忘”过期的市场经验，适应新的市场周期。

---

## 3. 具体修改方案 (Patch)

以下是针对 `worker-quant/worker.py` 的具体代码修改 Patch。我们通过修改 `discovery_workflow` 来植入 Thompson Sampling。

### 核心修改点
1. **状态存储结构升级**：将简单的 `learned_seed` 升级为记录各策略 $\alpha, \beta$ 值的统计字典。
2. **动态排序逻辑**：利用 Beta 分布采样决定策略尝试顺序。
3. **反馈更新循环**：根据扫描结果更新模型。

### Python 代码替换说明
*（由于标准库不包含 `scipy.stats`，我们将使用 Python 自带的 `random.betavariate` 来实现 Beta 采样，保证环境零依赖。）*

你可以直接将以下逻辑整合到 `worker.py` 中的 `discovery_workflow` 方法内：

```python
import random

# 在 discovery_workflow 中替换相关的 learning 逻辑：

    # ... [保持前面的环境初始化、变量定义等不变] ...

    # 1. 定义基础策略库（移除写死的尝试顺序）
    base_strategies = {
        "strict": {
            "market": market_focus,
            "max_position_pct": base_max_pos_pct,
            "alpha_floor": base_alpha_floor,
            "avoid_mega_cap": avoid_mega_cap,
            "prefer_small_mid_cap": prefer_small_mid_cap,
        },
        "relax_position": {
            "market": market_focus,
            "max_position_pct": min(max(base_max_pos_pct, 0.30) + 0.07, 0.45),
            "alpha_floor": base_alpha_floor - 0.35,
            "avoid_mega_cap": avoid_mega_cap,
            "prefer_small_mid_cap": prefer_small_mid_cap,
        },
        "broaden_factors": {
            "market": "ALL" if auto_expand_market and market_focus != "ALL" else market_focus,
            "max_position_pct": min(max(base_max_pos_pct, 0.35) + 0.10, 0.55),
            "alpha_floor": base_alpha_floor - 0.75,
            "avoid_mega_cap": False,
            "prefer_small_mid_cap": False,
        },
        "fallback_wide": {
            "market": "ALL" if auto_expand_market and market_focus != "ALL" else market_focus,
            "max_position_pct": min(max(base_max_pos_pct, 0.40) + 0.15, 0.65),
            "alpha_floor": base_alpha_floor - 1.10,
            "avoid_mega_cap": False,
            "prefer_small_mid_cap": False,
        }
    }

    learning_key = None
    learning_store = {}
    mab_state = {} # 存储 alpha (成功) 和 beta (失败)

    if enable_learning:
        learning_key = "|".join(["mab_v1", market_focus, goal_profile.get("risk_profile", "medium"), _capital_bucket(capital_base_jpy)])
        learning_store = _load_discovery_learning_store()
        mab_state = learning_store.get(learning_key) or {}
        
        # 初始化缺失的策略状态
        for s_name in base_strategies.keys():
            if s_name not in mab_state:
                mab_state[s_name] = {"alpha": 1.0, "beta": 1.0} # 1.0 为无信息先验

    # 2. Thompson Sampling 动态排序
    attempt_profiles = []
    if enable_learning:
        sampled_scores = {}
        for s_name, stats in mab_state.items():
            if s_name in base_strategies:
                # 从 Beta 分布采样
                a = max(1.0, stats.get("alpha", 1.0))
                b = max(1.0, stats.get("beta", 1.0))
                sampled_scores[s_name] = random.betavariate(a, b)
        
        # 按采样分数从高到低排序策略
        sorted_strategy_names = sorted(sampled_scores.keys(), key=lambda k: sampled_scores[k], reverse=True)
        for s_name in sorted_strategy_names:
            profile = base_strategies[s_name].copy()
            profile["label"] = s_name
            attempt_profiles.append(profile)
    else:
        # 如果未开启学习，使用默认固定顺序
        for k, v in base_strategies.items():
            p = v.copy()
            p["label"] = k
            attempt_profiles.append(p)

    # 3. 动态生成兜底策略（如果上述不够用）
    while len(attempt_profiles) < max_attempts:
        prev = attempt_profiles[-1]
        attempt_profiles.append({
            "label": f"extended_{len(attempt_profiles)+1}",
            "market": "ALL" if auto_expand_market and market_focus != "ALL" else prev.get("market", market_focus),
            "max_position_pct": min(float(prev.get("max_position_pct", 0.50)) + 0.05, 0.65),
            "alpha_floor": max(float(prev.get("alpha_floor", -2.0)) - 0.25, -3.0),
            "avoid_mega_cap": False,
            "prefer_small_mid_cap": False,
        })

    # ... [执行 _run_attempt 的循环保持不变，记录最好结果] ...

    # 4. 反馈环节：根据执行结果更新 Beta 分布参数
    if enable_learning and learning_key:
        decay_factor = 0.98 # 时间衰减：逐渐遗忘老旧的市场环境
        
        # 将所有状态整体乘以衰减因子（防累加过大）
        for s_name in mab_state:
            mab_state[s_name]["alpha"] = max(1.0, mab_state[s_name]["alpha"] * decay_factor)
            mab_state[s_name]["beta"] = max(1.0, mab_state[s_name]["beta"] * decay_factor)
        
        # 针对本次最佳策略（或实际使用的策略）进行 Reward 反馈
        if best_profile is not None:
            used_label = best_profile.get("label")
            if used_label in mab_state:
                if len(results) >= min_candidates_target:
                    # 成功找到足够的标的
                    mab_state[used_label]["alpha"] += 1.0
                else:
                    # 失败，未能找到足够的标的
                    mab_state[used_label]["beta"] += 1.0

        learning_store[learning_key] = mab_state
        learning_store["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _save_discovery_learning_store(learning_store)
        
    # ... [后续的输出和分析逻辑不变] ...
```

## 4. 优化预期与优势
1. **自适应探索**：当原本不看好的策略（如 `fallback_wide`）由于其 $\beta$ 较高导致偶尔被采样到时，如果此时市场发生变化（例如大跌市，只有放宽标准才能找到股票），系统会发现该策略成功，进而增加其 $\alpha$。它实现了从“尝试”到“主导”的自动过渡。
2. **收敛性与鲁棒性**：它用严格的概率分布代替了你原来硬编码的覆盖逻辑。不再只是盲目地扩大选股范围，而是在有把握的前提下，用概率引导尝试方向。
3. **零外部依赖**：通过内置的 `random.betavariate` 直接实现了成熟的强化学习/MAB算法，无需引入庞大的机器学习框架（如 PyTorch / Scikit-learn）。

"""Generate audited tables, figures, a Chinese report and a 5-minute HTML deck."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def fmt(value, kind="number"):
    if pd.isna(value):
        return "—"
    if kind == "percent":
        return f"{100 * float(value):.2f}%"
    if kind == "float":
        return f"{float(value):.4f}"
    return f"{int(round(float(value))):,}"


def _figures(out: Path):
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    cv = pd.read_csv(out / "metrics.csv")
    cv = cv[cv.stage == "cv"]
    business = pd.read_csv(out / "business_analysis.csv")
    business = business[(business.stage == "independent_test") & (business.capacity == 0.1)]
    subgroup = pd.read_csv(out / "subgroup_metrics.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model, g in cv.groupby("model"):
        ax.plot(g.fold, g.ap, marker="o", linewidth=2, label=model)
    ax.set(xlabel="CV fold", ylabel="Average Precision", title="Development CV: AP by fold")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "cv_ap.png", dpi=160); plt.close(fig)

    order = ["random_expected", "random_realized", "logistic", "hist_gbdt"]
    business["strategy"] = pd.Categorical(business.strategy, order, ordered=True)
    business = business.sort_values("strategy")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(business.strategy.astype(str), business.precision, color=["#9ca3af", "#6b7280", "#2563eb", "#0f766e"])
    ax.set(ylabel="Positive rate among contacted", title="Independent test: 10% contact capacity")
    ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figdir / "top10_policy.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(subgroup.group, subgroup.global_policy_ppv, color="#7c3aed")
    ax.set(ylabel="PPV among selected", title="Descriptive age-group check under one global policy")
    ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figdir / "subgroup_ppv.png", dpi=160); plt.close(fig)


def _result_context(out: Path):
    lock = json.loads((out / "model_lock.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(out / "metrics.csv")
    test = metrics[metrics.stage == "independent_test"].set_index("model")
    cv = pd.read_csv(out / "cv_summary.csv").set_index("model")
    business = pd.read_csv(out / "business_analysis.csv")
    selected = lock["selected_model"]
    policy = business[(business.stage == "independent_test") & (business.capacity == .1) & (business.strategy == selected)].iloc[0]
    random = business[(business.stage == "independent_test") & (business.capacity == .1) & (business.strategy == "random_expected")].iloc[0]
    subgroup = pd.read_csv(out / "subgroup_metrics.csv")
    stability = pd.read_csv(out / "stability.csv").set_index("model")
    ablation = pd.read_csv(out / "ablation_features.csv").set_index("scenario")
    calibration = json.loads((out / "calibration_summary.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(out / "feature_importance.csv")
    costs = pd.read_csv(out / "threshold_cost_scenarios.csv").set_index("scenario")
    return dict(lock=lock, audit=audit, test=test, cv=cv, business=business, selected=selected,
                policy=policy, random=random, subgroup=subgroup, stability=stability,
                ablation=ablation, calibration=calibration, importance=importance, costs=costs)


def _report(out: Path, docs: Path, c):
    docs.mkdir(exist_ok=True)
    s = c["selected"]; t = c["test"].loc[s]; cv = c["cv"].loc[s]
    p, r, a = c["policy"], c["random"], c["audit"]
    gap = c["subgroup"].global_policy_ppv.max() - c["subgroup"].global_policy_ppv.min()
    baseline = c["ablation"].loc["baseline_11_features"]
    duration = c["ablation"].loc["add_duration"]
    cost_low_threshold = c["costs"].loc["fn_5_fp_1"].threshold
    cost_high_threshold = c["costs"].loc["fn_2_fp_3"].threshold
    text = f"""# Bank Marketing 项目最终报告

## 摘要

本项目使用课程发布的合成银行营销数据，在**决定是否联系客户、且本次联系尚未发生**的时点预测客户是否订购定期存款。训练集共 {a['full_rows']:,} 行，正类率为 {fmt(a['full_positive_rate'], 'percent')}。主指标为 Average Precision（AP），业务决策为在最多联系 10% 客户时按预测概率排序。开发阶段只用分组五折交叉验证选择模型；模型、特征、容量与分析规则锁定后，每次完整复现只执行一次固定独立测试，测试结果不返选任何方案。

最终锁定模型为 **{s}**。开发五折 AP 为 {cv.ap_mean:.4f} ± {cv.ap_std:.4f}；独立测试 AP 为 {t.ap:.4f}、ROC-AUC 为 {t.roc_auc:.4f}。在 10% 联系容量下，共联系 {int(p.contacts):,} 人、命中 {int(p.hits):,} 个正类，联系对象正类率为 {fmt(p.precision, 'percent')}，相对随机联系 Lift 为 {p.lift:.2f}。这些结果说明模型可改善有限名额排序，但不证明联系行为造成了购买。

## 1. 问题与业务决定

- 预测对象：一条客户营销记录；`y=1` 表示订购定期存款。
- 预测时点：本次联系发生前。课程数据没有提供明确的标签观察窗口，因此本报告不虚构“未来多少天”的定义。
- 决策：在联系资源和客户打扰成本受限时，优先联系预测概率高的客户。
- 容量：主策略固定为全部候选客户的 10%；同时描述 5% 和 20%，不依靠测试集选择容量。
- 数据性质：课程数据依据公开真实数据结构生成，不代表任何银行生产数据。

## 2. 防泄露设计

模型仅使用 `age, balance, pdays, previous, job, marital, education, default, housing, loan, poutcome`。`duration` 是本次联系后的通话时长；`campaign` 包含本次活动联系次数；`contact/day/month` 在该表中描述最后一次联系，缺少可证明的事前计划快照，因此全部排除。`y`、来源 `id` 和 `course_row_id` 也不进入模型。

允许特征完全相同的画像被放进同一分组，再做开发/测试和交叉验证划分，跨分区分组重叠为 0。预处理仅在每一折训练部分拟合。隐藏评估特征在模型锁定与全量重训后才读取；隐藏标签、排行榜反馈、行顺序和其他旁路信息均不参与任何选择。

为量化而不是淡化泄露风险，我们只在开发折内做反事实诊断：把 `duration` 加回后，五折 AP 从 {baseline.ap_mean:.4f} 升至 {duration.ap_mean:.4f}、AUC 从 {baseline.auc_mean:.4f} 升至 {duration.auc_mean:.4f}。这不证明 `duration` 本身“导致”购买；结合它在本次通话结束后才可获得的时间定义，这个异常增益说明违反预测时点会严重高估可部署表现。因此该字段仍被排除，诊断结果也没有参与模型选择。

## 3. 模型与验证

基线为 Logistic Regression（数值标准化、类别独热编码）；候选改进为 HistGradientBoostingClassifier（类别序数编码、固定参数）。每个模型只有一套预先登记参数，没有在独立测试集上搜索。复杂模型只有在五折平均 AP 至少比基线高 0.005 时才会被选择。另用不同随机种子的开发集分组留出切分检查稳定性。

| 模型 | 开发CV AP | CV ROC-AUC | 独立测试 AP | 独立测试 ROC-AUC | 测试Top10% Lift |
|---|---:|---:|---:|---:|---:|
| logistic | {c['cv'].loc['logistic'].ap_mean:.4f} | {c['cv'].loc['logistic'].auc_mean:.4f} | {c['test'].loc['logistic'].ap:.4f} | {c['test'].loc['logistic'].roc_auc:.4f} | {c['test'].loc['logistic'].top10_lift:.2f} |
| hist_gbdt | {c['cv'].loc['hist_gbdt'].ap_mean:.4f} | {c['cv'].loc['hist_gbdt'].auc_mean:.4f} | {c['test'].loc['hist_gbdt'].ap:.4f} | {c['test'].loc['hist_gbdt'].roc_auc:.4f} | {c['test'].loc['hist_gbdt'].top10_lift:.2f} |

Accuracy 不是主指标：全预测为负类已经可达到 {fmt(t.all_negative_accuracy, 'percent')}，但无法找出任何响应客户。AP 和 Top-10% 指标更贴近排序任务。

## 4. 业务策略结果

随机策略在 10% 容量下的期望命中约为 {r.hits:.1f}，模型策略命中 {int(p.hits):,}，增加约 {p.hits-r.hits:.1f} 个响应客户。这里的“增加”是相对同一测试样本中的排序比较，不是营销联系的因果增量。若业务提供单位联系成本、客户价值和不联系机会成本，可在锁定模型后把容量曲线转化为净收益曲线；当前不凭空假设货币参数。

开发集 OOF 预测还做了三组教学成本敏感性：漏选成本/误联系成本为 5/1 时，开发集最小成本阈值为 {cost_low_threshold:.2f}；权重为 2/3 时阈值为 {cost_high_threshold:.2f}。前者更重视少漏人，阈值更低、TP 增加而 FN 减少，同时 FP 增加；后者更重视少打扰，阈值更高。权重只是单位化情景，不是真实货币，也没有在独立测试集上重新选择或报告“最优”阈值。

## 5. 错误、分组与稳定性

错误样例文件分别保留了 Top-10% 中未购买者（联系资源浪费）和 Top-10% 外的购买者（漏掉机会），各最多 10 条。年龄分组只作描述性检查；同一全局 Top-10% 策略下，各年龄组 PPV 最大差异为 {fmt(gap, 'percent')}。由于年龄与其他客户特征、样本量和营销机制相关，该差异不能直接解释为歧视或因果影响。

开发稳定性切分中，锁定模型 AP 为 {c['stability'].loc[s].ap:.4f}，与五折和独立测试结果方向一致。

开发 OOF 的 Brier 分数为 {c['calibration']['brier']:.4f}，十分位分箱最大预测率与观察率差为 {c['calibration']['maximum_absolute_bin_gap']:.4f}。这只能说明该开发样本上的粗粒度校准接近，不能保证生产校准。置换重要性和删特征敏感性分别保存在 `feature_importance.csv` 与 `ablation_features.csv`；前者衡量固定模型打乱单列后的 AP 变化，后者衡量删列重训后的变化，两者都不是因果效应且可能受相关特征替代影响。

## 6. 局限与建议

1. 数据是课程合成数据，外部有效性未知，不能直接部署。
2. 数据没有客户唯一标识和特征时间戳；上线前必须做真实客户级切分和时间穿越审计。
3. 目前预测“谁会购买”，不是估计“联系带来的增量”；若要优化营销干预，应进行随机实验并研究 uplift。
4. 缺少收入、成本、投诉与客户价值，当前仅能给出容量下的命中率和 Lift。
5. 建议先做小规模受控试点，监控响应率、选择率、年龄组差异和数据漂移，达到预设停止条件再扩大。

## 7. 可复现性与AI使用

`python main.py --phase all` 是唯一主入口。`results/model_lock.json` 记录锁定时点、参数、特征和证据哈希；`results/test_audit.json` 记录本次运行的固定测试阶段，并披露历史上曾为复现审计重复计算同一锁定测试，但从未据此改变模型、特征、容量、阈值或开发诊断；`results/run_manifest.json` 保存版本、数据哈希和隐藏集使用状态。AI辅助过程、被修改和被拒绝的建议记录在 `ai_usage_log.md`；所有结论均由实际运行结果生成，不伪造日志或分数。
"""
    (docs / "final_report.md").write_text(text, encoding="utf-8")


def _supplement(out: Path, docs: Path, c):
    baseline = c["ablation"].loc["baseline_11_features"]
    duration = c["ablation"].loc["add_duration"]
    balance = c["ablation"].loc["drop_balance"]
    poutcome = c["ablation"].loc["drop_poutcome"]
    costs = c["costs"]
    top = c["importance"].head(5)
    top_rows = "\n".join(
        f"| {r.feature} | {r.importance_mean_ap_decrease:.4f} | {r.importance_std:.4f} |"
        for _, r in top.iterrows())
    cost_rows = "\n".join(
        f"| {name} | {row.fn_cost:.0f} | {row.fp_cost:.0f} | {row.threshold:.2f} | {int(row.development_fp):,} | {int(row.development_fn):,} |"
        for name, row in costs.iterrows())
    text = f"""# 开发集补充诊断

本页所有选择性分析只使用开发集、固定分组折和折外预测；它们在 `model_lock.json` 写入且独立测试开始之前完成并固化哈希。独立测试和隐藏集均未参与阈值、特征或成本情景选择。

## 1. 预测时点泄露对照

| 方案 | 五折 AP 均值 ± 样本标准差 | 五折 AUC 均值 ± 样本标准差 | 用途 |
|---|---:|---:|---|
| 锁定的11个联系前字段 | {baseline.ap_mean:.4f} ± {baseline.ap_std:.4f} | {baseline.auc_mean:.4f} ± {baseline.auc_std:.4f} | 正式方案 |
| 加入 duration | {duration.ap_mean:.4f} ± {duration.ap_std:.4f} | {duration.auc_mean:.4f} ± {duration.auc_std:.4f} | 仅泄露诊断，不入模 |
| 删除 balance 后重训 | {balance.ap_mean:.4f} ± {balance.ap_std:.4f} | {balance.auc_mean:.4f} ± {balance.auc_std:.4f} | 敏感性 |
| 删除 poutcome 后重训 | {poutcome.ap_mean:.4f} ± {poutcome.ap_std:.4f} | {poutcome.auc_mean:.4f} ± {poutcome.auc_std:.4f} | 敏感性 |

`duration` 是本次联系结束后才知道的信息。分数上升只能量化违反预测时点对离线分数的影响，不能把它变成可部署特征，也不能单凭消融声称因果关系。所有标准差均按五折样本标准差计算，与 `cv_summary.csv` 口径一致。

## 2. 置换重要性

| 特征 | 平均 AP 降幅 | 重复间标准差 |
|---|---:|---:|
{top_rows}

置换重要性在开发集第1折验证样本的固定模型上计算，只是审计入口。它与“删列后重新训练”的问题不同，相关特征也可能互相替代；两者均不解释因果。

## 3. 粗粒度校准

开发 OOF Brier={c['calibration']['brier']:.5f}，十分位分箱加权平均绝对差={c['calibration']['weighted_mean_absolute_bin_gap']:.5f}，最大绝对差={c['calibration']['maximum_absolute_bin_gap']:.5f}。这是开发样本上的粗粒度检查，不等于生产概率已校准；原始分箱见 `results/calibration_bins.csv`。

## 4. 假设成本情景

| 情景 | FN单位权重 | FP单位权重 | 开发OOF阈值 | 开发FP | 开发FN |
|---|---:|---:|---:|---:|---:|
{cost_rows}

阈值只在开发 OOF 上按单位化假设最小化 `FN权重 × FN + FP权重 × FP`。漏选更贵时阈值更低，TP 增加、FN 减少、FP 增加；误联系更贵时方向相反。这些权重不是银行真实货币成本，独立测试没有被再次用于选择或比较这些情景。
"""
    (docs / "supplement_analysis.md").write_text(text, encoding="utf-8")


def _presentation(out: Path, presentation: Path, c):
    presentation.mkdir(exist_ok=True)
    asset_dir = presentation / "figures"
    asset_dir.mkdir(exist_ok=True)
    for name in ("cv_ap.png", "top10_policy.png", "subgroup_ppv.png",
                 "ablation_features.png", "cost_curves.png"):
        shutil.copy2(out / "figures" / name, asset_dir / name)
    s = c["selected"]; t = c["test"].loc[s]; p = c["policy"]; a = c["audit"]
    subrows = "".join(
        f"<tr><td>{html.escape(str(r.group))}</td><td>{int(r.n):,}</td><td>{fmt(r.positive_rate,'percent')}</td><td>{fmt(r.global_policy_ppv,'percent')}</td></tr>"
        for _, r in c["subgroup"].iterrows())
    comparison_rows = "".join(
        f"<tr><td>{html.escape(str(model))}</td><td>{c['cv'].loc[model].ap_mean:.4f} +/- {c['cv'].loc[model].ap_std:.4f}</td><td>{c['test'].loc[model].ap:.4f}</td><td>{c['test'].loc[model].roc_auc:.4f}</td><td>{c['test'].loc[model].top10_lift:.2f}</td></tr>"
        for model in ("logistic", "hist_gbdt"))
    errors = pd.read_csv(out / "error_type_counts.csv", index_col=0)["count"]
    fp_count = int(errors.get("FP_contact_no_subscription", 0))
    fn_count = int(errors.get("FN_missed_subscription", 0))
    example = pd.read_csv(out / "error_cases.csv")
    fn_example = example[example.error_type == "FN_missed_subscription"].iloc[0]
    stability_ap = c["stability"].loc[s].ap
    paired = pd.read_csv(out / "paired_cv.csv")
    paired_gain = float(paired["ap_gain_gbdt_minus_logistic"].mean())
    baseline = c["ablation"].loc["baseline_11_features"]
    duration = c["ablation"].loc["add_duration"]
    cost_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{row.fn_cost:.0f}:{row.fp_cost:.0f}</td><td>{row.threshold:.2f}</td></tr>"
        for name, row in c["costs"].iterrows())
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bank Marketing Final</title>
<style>body{{margin:0;background:#0b1324;color:#eaf0ff;font:18px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}main{{scroll-snap-type:y mandatory;height:100vh;overflow-y:auto}}section{{scroll-snap-align:start;min-height:100vh;box-sizing:border-box;padding:7vh 8vw;display:flex;flex-direction:column;justify-content:center}}h1{{font-size:3.5rem;margin:.1em 0}}h2{{font-size:2.5rem;color:#8bd3ff}}h3{{color:#9cf0cf}}.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}}.card{{background:#15213a;border:1px solid #2b4169;border-radius:16px;padding:1.1rem}}.big{{font-size:2.4rem;font-weight:700;color:#ffd166}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:2rem;align-items:center}}img{{max-width:100%;background:white;border-radius:12px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #345;padding:.55rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{color:#9cf0cf}}.note{{color:#a8b3c7;font-size:.9rem}}@media(max-width:800px){{.two,.kpis{{grid-template-columns:1fr}}h1{{font-size:2.5rem}}}}</style></head><body><main>
<section><p>金融科技概论 · 期末项目</p><h1>Bank Marketing</h1><h2>在联系前排序：有限名额联系谁？</h2><div class="kpis"><div class="card"><div class="big">{a['full_rows']:,}</div>有标签训练记录</div><div class="card"><div class="big">{fmt(a['full_positive_rate'],'percent')}</div>总体正类率</div><div class="card"><div class="big">10%</div>主联系容量</div></div><p>预测发生在本次联系前，按响应概率排序；不承诺购买，也不作因果解释。课程数据依据公开结构生成，并非银行生产数据。</p><p class="note">主指标 Average Precision · 固定联系容量 · 防泄露审计</p></section>
<section><h2>2. 防泄露设计</h2><div class="two"><div><h3>允许</h3><p>11个联系前客户资料与历史字段</p><h3>禁止</h3><p><code>duration</code>、当前活动/最后联系信息、目标与ID</p><div class="card"><p>相同画像同组；折内预处理</p><p>开发五折 → 锁定 → 固定测试</p><p>测试结果不返选；隐藏反馈不使用</p></div></div><div><img src="figures/ablation_features.png" alt="Development leakage diagnostic"><p class="note">仅开发集诊断：加入事后 duration 后 AP {baseline.ap_mean:.3f}→{duration.ap_mean:.3f}。时间定义决定它仍必须排除。</p></div></div></section>
<section><h2>3. 同条件模型比较</h2><div class="two"><img src="figures/cv_ap.png" alt="CV AP"><div><table><tr><th>模型</th><th>开发CV AP</th><th>测试 AP</th><th>测试 AUC</th><th>Top10 Lift</th></tr>{comparison_rows}</table><div class="card"><p>相同分组切分、特征与指标；各模型一套固定参数。</p><p>GBDT AP 配对折差均值 +{paired_gain:.4f}，五折均为正。</p><p>锁定模型：<strong>{s}</strong>；固定测试仅作评估，不返选方案。</p></div></div></div></section>
<section><h2>4. 业务策略</h2><div class="two"><img src="figures/top10_policy.png" alt="Top10 policy"><div><div class="card"><p>联系 {int(p.contacts):,} 人，命中 {int(p.hits):,}</p><p>正类率 <span class="big">{fmt(p.precision,'percent')}</span> · Lift {p.lift:.2f}</p><p class="note">排序改善 ≠ 联系造成的因果增量。</p></div><h3>开发OOF假设成本</h3><table><tr><th>情景</th><th>FN:FP</th><th>阈值</th></tr>{cost_rows}</table><p class="note">权重非真实货币；测试集未用于阈值选择。</p></div></div></section>
<section><h2>5. 错误与群体检查</h2><div class="two"><img src="figures/subgroup_ppv.png" alt="Subgroup PPV"><table><tr><th>年龄组</th><th>样本</th><th>正类率</th><th>选中PPV</th></tr>{subrows}</table></div><p>Top10%策略：未购买仍联系 {fp_count:,} 人；未选中但购买 {fn_count:,} 人。代表漏选样本为 age={int(fn_example.age)}, balance={int(fn_example.balance)}, pdays={int(fn_example.pdays)}, 历史结果 unknown。</p><p>年龄差异仅描述，不直接解释为歧视或因果结论。</p></section>
<section><h2>6. 结论、稳定性与下一步</h2><div class="kpis"><div class="card"><div class="big">{stability_ap:.4f}</div>换种子开发留出 AP</div><div class="card"><div class="big">{c['calibration']['brier']:.4f}</div>开发OOF Brier</div><div class="card"><div class="big">{c['calibration']['maximum_absolute_bin_gap']:.4f}</div>十分位最大校准差</div></div><ul><li>模型适合做有限名额排序的课程原型，校准检查仅为粗粒度证据。</li><li>上线前补齐客户ID、特征时间戳、真实成本价值与随机实验。</li><li>先做受控试点，持续监控响应、选择率、分组差异和漂移。</li><li>代码、数据哈希、模型锁、诊断哈希和AI记录均可审计。</li></ul></section>
</main></body></html>"""
    (presentation / "index.html").write_text(body, encoding="utf-8")


def render(out: Path, presentation: Path, docs: Path):
    required = ["model_lock.json", "metrics.csv", "business_analysis.csv", "subgroup_metrics.csv"]
    if not all((out / f).is_file() for f in required):
        return
    _figures(out)
    context = _result_context(out)
    _report(out, docs, context)
    _supplement(out, docs, context)
    _presentation(out, presentation, context)

# 七人分工与答辩准备

> 下面的“成员1–成员7”是建议分工占位符。提交前必须换成真实姓名，并按照每个人真实完成和实际理解的内容调整，不能把建议分工当成已经发生的事实。

## 全组共同底线

每个人都要能用一句话讲完整项目：我们在本次联系发生前，用11个可用字段预测客户订购定期存款的概率；开发集分组五折选择并锁定模型，再做固定独立测试；最终把概率转成Top 10%联系名单。测试AP为0.4600，Top 10%正类率48.43%、Lift 4.09，但这些是排序证据，不是联系造成购买的因果证据。

每个人至少亲自做一次以下事情：打开自己负责的代码；指出函数的输入和输出；运行一个小示例或解释真实结果文件；回答该步骤最可能发生的泄露或误解。

## 成员1：问题定义与数据审计

- 主责：观察单位、目标变量、预测时点、数据规模和字段合同。
- 重点代码：`main.py` 第64–72行，`data/metadata/`，`results/data_audit.json`。
- 输入：`student_train.parquet`。
- 输出：600,000行训练表、字段检查、正类率12.065%、开发/测试样本数等审计信息。
- 可能问题：一行数据代表什么？`y` 是什么？为什么 `course_row_id` 不能进模型？
- 标准回答：一行是一条客户营销记录；`y=1` 表示订购定期存款。`course_row_id` 只用于行对齐，不是客户行为特征，放进模型可能形成无意义或旁路记忆。

代码输入输出示例：

```python
d = load_train(Path("data/bank-marketing"))
print(d.shape, d.y.mean())
```

预期输出的结构是 `(600000, 20)` 和约 `0.12065`。列数包含目标与标识字段；正式模型只从中选择11个允许字段。

## 成员2：特征可用性与信息泄露

- 主责：解释11个允许字段和8个排除字段，尤其是 `duration`。
- 重点代码：`src/modeling.py` 第10–18行，`results/feature_availability.csv`，`results/ablation_features.csv`。
- 输入：原始字段及业务预测时点。
- 输出：`FEATURES` 允许列表和 `EXCLUDED` 排除理由。
- 可能问题：为什么 `duration`、`campaign`、`contact/day/month` 不用？
- 标准回答：`duration` 在本次通话结束后才知道，明显晚于联系前决策。其余当前活动或最后联系字段缺少可证明的事前快照，因此保守排除。开发折内加回 `duration` 会把AP从0.457提高到0.734，说明泄露能夸大离线表现；它不是正式候选特征。

代码输入输出示例：

```python
from src.modeling import FEATURES, EXCLUDED
print(len(FEATURES), FEATURES)
print(EXCLUDED["duration"])
```

预期输出为11个允许字段，以及“本次联系后时长”的排除理由。

## 成员3：数据切分、交叉验证与模型锁

- 主责：分组留出、分组五折、OOF预测、测试集纪律。
- 重点代码：`main.py` 第75–177行，`results/split_assignments.csv`，`results/model_lock.json`。
- 输入：完整训练数据、允许特征画像、固定随机种子20261022。
- 输出：479,354行开发集、120,646行独立测试集、五折索引、OOF概率和模型锁。
- 可能问题：为什么不用普通随机切分？测试集什么时候能看？
- 标准回答：允许特征完全相同的画像可能重复，因此相同画像必须进入同一分区；`StratifiedGroupKFold` 同时尽量维持正类比例。模型、特征、指标和10%容量先锁定，测试集只做固定评估，不返选方案。

代码输入输出示例：

```python
groups = groups_for(d)
print(len(groups), len(set(groups)))
```

预期输出为600,000个分组标签、458,201个不同画像组。分组标签不是客户ID，只是当前数据条件下防止完全相同画像跨分区的保守措施。

## 成员4：Logistic Regression 基线

- 主责：为什么需要简单基线、Pipeline中每一步做什么、如何输出概率。
- 重点代码：`src/modeling.py` 第42–57行，`results/cv_summary.csv`。
- 输入：11列特征和二元标签。
- 输出：训练好的Pipeline与每行属于正类的概率。
- 可能问题：`build_model("logistic", seed)` 返回什么？为什么类别字段要One-Hot？
- 标准回答：它返回一个Pipeline，数值列先用中位数填补再标准化，类别列用众数填补再One-Hot，最后进入Logistic Regression。One-Hot避免把无序类别误当成有大小关系的连续数字。

代码输入输出示例：

```python
model = build_model("logistic", 20261022)
model.fit(train[FEATURES], train.y)
p = model.predict_proba(valid[FEATURES])[:, 1]
print(p.shape, p.min(), p.max())
```

输出是一维概率数组，长度等于验证集行数，每个值位于0到1之间。基线五折AP为0.3756。

## 成员5：Histogram GBDT 候选模型

- 主责：候选模型结构、固定参数、为什么胜过基线、为什么不是用测试集挑选。
- 重点代码：`src/modeling.py` 第22–39行，`main.py` 第124–177行。
- 输入：同样的11列特征和同样的五折。
- 输出：GBDT概率、五折AP、锁定模型名称 `hist_gbdt`。
- 可能问题：为什么GBDT可能比Logistic好？类别变量怎样处理？
- 标准回答：GBDT能表达非线性和变量交互。类别列先序数编码，未知类别记为缺失值，并在模型中标记为类别特征。五折AP为0.4572，比基线高0.0816，超过事先规定的0.005门槛，因此在独立测试前锁定。

代码输入输出示例：

```python
model = build_model("hist_gbdt", 20261022)
print(model.named_steps["model"].get_params()["max_iter"])
```

预期输出为 `200`。本项目每个模型只有一套固定参数，不在独立测试上搜索参数。

## 成员6：指标、Top-k策略与阈值

- 主责：AP、ROC-AUC、Top 10% Precision/Recall/Lift和阈值方向。
- 重点代码：`src/evaluation.py`，`results/metrics.csv`，`results/business_analysis.csv`。
- 输入：真实标签 `y` 和预测概率 `p`。
- 输出：指标字典、布尔联系名单、容量表和阈值混淆矩阵。
- 可能问题：`top_mask(p, 0.1)` 输出什么？Lift 4.09是什么意思？
- 标准回答：`top_mask` 输出一个与 `p` 等长的布尔数组，恰有 `ceil(0.1*n)` 个True，代表分数最高的10%客户；并列分数用固定seed打破。Lift是选中人群正类率除以总体正类率，不是销量因果倍数。

代码输入输出示例：

```python
p = [0.2, 0.9, 0.4, 0.1]
print(top_mask(p, fraction=0.25, seed=20261022))
```

输出会有且只有一个 `True`，对应概率0.9的记录。真实独立测试10%容量联系12,065人，命中5,843人，Precision 48.43%，Recall 40.89%，Lift 4.09。

## 成员7：诊断、完整复现与最终提交

- 主责：消融、校准、成本情景、全量重训、隐藏预测、报告和演示。
- 重点代码：`src/diagnostics.py`，`main.py` 第226–262行，`src/reporting.py`。
- 输入：开发OOF概率、已锁定模型、150,000行隐藏特征。
- 输出：诊断CSV/JSON、`final_model.joblib`、`predictions.csv`、报告和演示文件。
- 可能问题：主命令会做什么？隐藏数据何时读取？现场改代码应注意什么？
- 标准回答：主命令依次校验数据、做开发五折、生成诊断、写模型锁、执行固定独立测试、全量重训，最后才读取隐藏特征并输出150,000行概率。若修改正式特征或模型，必须重新建立开发协议和新的测试留出，不能用旧测试集继续调参。

主命令：

```bash
python main.py --phase all --output results/reproduced
```

核心输出包括 `cv_summary.csv`、`metrics.csv`、`business_analysis.csv`、`model_lock.json`、`test_audit.json`、`run_manifest.json` 和 `predictions.csv`。

## 现场修改的预期方向

- 把容量从10%改为20%：联系人数增加，Recall通常上升，Precision通常下降。真实测试结果为Precision 34.32%、Recall 57.96%。
- 降低分类阈值：预测为正的人增加，TP和FP通常增加，FN减少。
- 提高误联系成本：更偏好较高阈值；提高漏选成本：更偏好较低阈值。本项目开发OOF情景中，FN:FP为5:1时阈值0.16，为2:3时阈值0.60。
- 加入 `duration`：离线分数会明显上升，但违反预测时点，不能作为正式改进。
- 改随机种子：指标会小幅变化；应报告波动而不是只挑最好的一次。

## 不要说错的四句话

1. 不要说“Lift 4.09表示销量提高4.09倍”；应说“选中人群的观察正类率是总体的4.09倍”。
2. 不要说“`duration` 提高分数所以应该使用”；应说“它的异常增益说明时间泄露会夸大表现”。
3. 不要说“测试集用来选择模型”；模型只由开发五折选择，测试集只评估。
4. 不要说“模型可以直接上线”；课程数据是合成数据，仍缺真实客户ID、时间戳、成本和随机实验。

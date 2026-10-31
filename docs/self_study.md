# 项目自学条目：Average Precision 与 Top-k Lift

## 概念

Average Precision（AP）概括 Precision–Recall 曲线，关注模型是否把正类排在靠前位置。Top-k Lift 衡量在固定容量 `k` 下，模型选中人群的正类率相对全体正类率提高了多少倍。

## 输入与假设

- 输入：真实标签 `y` 与模型给出的连续概率分数。
- 正类固定为 `y=1`，即客户订购定期存款。
- 主容量 `k=10%` 在查看独立测试结果前固定。
- 分数并列时使用固定随机种子打破并列，不使用标签、ID 或行顺序。

## 计算

将客户按分数从高到低排序。AP 是各个正类位置上的 Precision 加权平均；本项目直接使用 scikit-learn 的 `average_precision_score`。

手算例子：标签为 `[1, 0, 1, 0]`，分数排序后正类出现在第1和第3位。两处 Precision 分别为 `1/1` 和 `2/3`，所以 AP = `(1 + 2/3) / 2 = 0.8333`。若总体正类率是 `2/4=0.5`，Top-50% 选中前两行只有1个正类，Precision=`1/2`，Lift=`0.5/0.5=1`。

可复核的最小代码例子：

```python
from sklearn.metrics import average_precision_score
average_precision_score([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
# 0.8333333333333333
```

```text
Top-10% Precision = Top-10% 中的正类数 / Top-10% 联系人数
Top-10% Lift = Top-10% Precision / 全部样本正类率
Top-10% Recall = Top-10% 中的正类数 / 全部正类数
```

## 项目结果

- 独立测试 AP：0.4600。
- 测试样本：120,646；正类：14,289；总体正类率：11.84%。
- 联系容量：12,065 人（10% 向上取整）。
- 命中正类：5,843；Top-10% Precision：48.43%；Recall：40.89%；Lift：4.09。

## 边界

AP 与 Lift 评价排序，不说明概率已经完全校准，也不说明营销联系导致了购买。Lift 会随样本正类率、时间、渠道和容量改变，不能脱离分母比较。真实业务还需成本价值、随机实验和时间外验证。

## 参考

- scikit-learn 用户指南：[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- 课程参考入口：[`项目自学入口`](https://fintech-ml.netlify.app/references#project-self-study)

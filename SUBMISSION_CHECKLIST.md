# 最终提交清单

## 已完成

- [x] 可复现主程序与依赖文件
- [x] Logistic Regression 可解释基线
- [x] Histogram GBDT 候选改进
- [x] 分组五折开发验证与锁定后的固定独立本地测试
- [x] AP、ROC-AUC、Top-5/10/20%、Lift、阈值分析
- [x] 至少10个错误样例与错误类型
- [x] 年龄组差异检查与稳定性检查
- [x] 开发集 duration 泄露反事实、删特征敏感性与置换重要性
- [x] 开发 OOF 粗粒度校准和假设成本阈值（不使用测试集选择）
- [x] 模型锁、数据哈希、测试审计与隐藏数据审计
- [x] 150,000行隐藏特征预测 `results/predictions.csv`
- [x] 中文最终报告 `docs/final_report.md`
- [x] 5分钟HTML展示 `presentation/index.html`
- [x] 6页主讲 + 4页答辩附录的PowerPoint `presentation/Bank-Marketing-Defense.pptx`
- [x] AI使用记录、自学条目、补充诊断、答辩速查和七人分工手册
- [x] 自动化测试通过（最终打包前重新执行）

## 提交前需要学生补充

- [ ] 在提交入口填写姓名、学号、班级与组员（包内不虚构个人信息）。
- [ ] 若为小组，写明每个人的真实分工；不要让AI虚构贡献。
- [ ] 将PPT中的 `成员1–成员7` 替换为真实姓名，并核对每个人实际参与的内容。
- [ ] 向老师确认提交平台、截止时间、仓库是否必须公开。
- [ ] 将项目放进你自己的 Git 仓库；不要提交 `.venv/` 或 Parquet 数据。
- [ ] 用PowerPoint打开 `presentation/Bank-Marketing-Defense.pptx`，按前6页在5分钟内至少完整讲两遍。
- [ ] 七名成员分别用 `docs/team_roles_and_defense.md` 演练自己的输入、代码动作、输出和追问。
- [ ] 能现场解释并修改 `main.py`、`src/modeling.py` 和 `src/evaluation.py`。
- [ ] 如果老师要求特定预测列名或文件名，只改提交副本格式，不用评估反馈反调模型。

## 建议提交内容

提交代码仓库链接，并按老师指定渠道附上展示文件或链接。仓库保留 `results/predictions.csv` 和小型结果表/图片；原始 Parquet 数据与模型二进制已由 `.gitignore` 排除。若平台限制仓库体积，可不提交 `development_oof.csv`、`split_assignments.csv` 和 `local_test_predictions.csv`，但应保留它们的生成代码与 `model_lock.json` 中的哈希说明。

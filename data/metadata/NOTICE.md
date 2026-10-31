# 来源与使用说明

本仓库是为课程教学整理的固定版本数据包。公开的是课程抽样、字段整理和项目切分后的 Parquet 文件，不是上游数据集的完整镜像。

## 来源

- California Housing：课程使用 scikit-learn / California Housing 的固定课堂样本。原始数据说明见 [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)。
- UCI 信用卡违约：来源为 [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients)。
- `loan-approval`：来源为 Kaggle [Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10/data)，Kaggle 页面标注 CC BY 4.0。
- `loan-payback`：来源为 Kaggle [Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11/data)，请同时遵守 Kaggle 竞赛规则和数据页面条款。
- `bank-marketing`：来源为 Kaggle [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/data)，Kaggle 页面标注 Apache 2.0。
- `insurance-premium`：来源为 Kaggle [Regression with an Insurance Dataset](https://www.kaggle.com/competitions/playground-series-s4e12/data)，Kaggle 页面标注 Apache 2.0。
- `lending-club`：来源为 Kaggle [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)，数据页面标注 CC0 Public Domain。

## 使用边界

本仓库的课程整理文件按各上游数据页面和规则使用。转载、再发布或用于课程以外的用途时，请自行核对当前上游条款并保留来源说明。课程仓库不对上游数据条款作额外授权。

Home Credit Stability 数据没有放在本仓库中。官方规则写明该竞赛数据只可用于参加竞赛及 Kaggle 论坛，并禁止向未参加者传输、复制、公开或再分发。请不要从其他地方下载后再上传到本仓库。

本仓库的清单、字段说明、质量记录和课程文字说明由任课教师维护。数据按现状提供，不代表任何金融机构的生产数据，也不构成信贷、保险或营销决策建议。

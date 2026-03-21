import numpy as np
import pandas as pd

# 固定随机种子，保证数据可复现
np.random.seed(42)

# 参数
n_samples = 2000          # 总样本数
# 购买概率公式：log_odds = 0.2*duration + 0.001*past_spend - 2.5
# 这样购买比例大约在 30% 左右

# 生成特征
duration = np.random.uniform(0, 30, n_samples)          # 浏览时长 0~30 分钟
past_spend = np.random.uniform(0, 2000, n_samples)      # 历史消费 0~2000 元

# 计算购买概率
log_odds = 0.2 * duration + 0.001 * past_spend - 2.5
prob_purchase = 1 / (1 + np.exp(-log_odds))

# 生成购买标签（伯努利试验）
purchased = np.random.binomial(1, prob_purchase)

# 生成购买金额（仅对购买样本）
amount = np.full(n_samples, np.nan)
purchase_idx = purchased == 1
# 金额与特征线性相关，加噪声
amount[purchase_idx] = (0.5 * duration[purchase_idx] +
                        0.1 * past_spend[purchase_idx] +
                        np.random.normal(0, 50, size=purchase_idx.sum()))

# 创建 DataFrame
df = pd.DataFrame({
    'duration': duration,
    'past_spend': past_spend,
    'purchased': purchased,
    'amount': amount
})

# 保存为 CSV 文件（不保存索引）
df.to_csv('ecommerce_data.csv', index=False)

print(f"数据集已生成，共 {n_samples} 条记录")
print(f"购买比例: {purchased.mean():.2%}")
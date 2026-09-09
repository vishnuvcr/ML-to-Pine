import joblib
import math
from sklearn.ensemble import VotingClassifier

# Load the trained model
model = joblib.load("universal_strategy.joblib")

# 1. Extract the HistGradientBoosting model from the VotingClassifier
if isinstance(model, VotingClassifier):
    tree_model = model.named_estimators_['hgb']
else:
    tree_model = model

feature_names = list(tree_model.feature_names_in_)
baseline = float(tree_model._baseline_prediction.item())
predictors = tree_model._predictors

pine_code = []
pine_code.append("//@version=6")
pine_code.append('strategy("Lossless GBDT Strategy (NSE)", overlay=true, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, process_orders_on_close=true)')
pine_code.append("")

# 2. EXACT FEATURE RECONSTRUCTION (Including New Volume & ATR Features)
pine_code.append("// 1. FEATURES")
pine_code.append("return_1d = (close - close[1]) / close[1]")
pine_code.append("volatility_20d = ta.stdev(return_1d, 20)")
pine_code.append("sma_20_ratio = (close / ta.sma(close, 20)) - 1.0")
pine_code.append("sma_50_ratio = (close / ta.sma(close, 50)) - 1.0")
pine_code.append("rsi_14 = ta.rsi(close, 14)")
pine_code.append("vol_sma_20 = ta.sma(volume, 20)")
pine_code.append("volume_ratio_20 = volume / (vol_sma_20 + 1e-9)")
pine_code.append("atr_14 = ta.atr(14)")
pine_code.append("atr_ratio = atr_14 / close")
pine_code.append("")

def render_node(nodes, idx):
    node = nodes[idx]
    if node["is_leaf"]:
        return f"{node['value']:.8f}"
    feat = feature_names[node["feature_idx"]]
    thresh = node["num_threshold"]
    left_repr = render_node(nodes, node["left"])
    right_repr = render_node(nodes, node["right"])
    return f"({feat} <= {thresh:.8f} ? {left_repr} : {right_repr})"

pine_code.append("// 2. COMPILED TREE ENSEMBLE (HGB PROXY)")
chunk_size = 10
# Cap total trees to 85 to stay under TradingView's 80,000 token limit
MAX_TREES = 85
total_trees = min(len(predictors), MAX_TREES)

chunk_calls = []
for c_idx in range(0, total_trees, chunk_size):
    chunk_trees = predictors[c_idx:c_idx + chunk_size]
    func_name = f"calc_chunk_{c_idx // chunk_size}"
    chunk_calls.append(f"{func_name}()")
    
    pine_code.append(f"{func_name}() =>")
    pine_code.append("    float score = 0.0")
    for ti, tree in enumerate(chunk_trees):
        expr = render_node(tree[0].nodes, 0)
        pine_code.append(f"    score += {expr}")
    pine_code.append("    score")
    pine_code.append("")

pine_code.append("// 3. INFERENCE & EXECUTION")
pine_code.append(f"float baseline = {baseline:.8f}")
pine_code.append(f"float raw_margin = baseline + {' + '.join(chunk_calls)}")
pine_code.append("float prob = 1.0 / (1.0 + math.exp(-raw_margin))")
pine_code.append("")

# Lowered threshold to match the balanced ensemble logic
pine_code.append("input_threshold = input.float(0.52, 'Buy Probability Threshold', minval=0.40, maxval=0.95, step=0.01)")
pine_code.append("long_condition = prob >= input_threshold and barstate.isconfirmed")
pine_code.append("")

pine_code.append("if (long_condition)")
pine_code.append('    strategy.entry("Long", strategy.long)')
pine_code.append("")

# Implement the dynamic 1.5:1 Risk/Reward Target
pine_code.append('// Dynamic SL and TP based on ATR')
pine_code.append('sl_price = close - (atr_14 * 1.0)')
pine_code.append('tp_price = close + (atr_14 * 1.5)')
pine_code.append('if (strategy.position_size > 0)')
pine_code.append('    strategy.exit("Exit Long", "Long", limit=tp_price, stop=sl_price)')
pine_code.append("")

pine_code.append('plot(prob, "Model Probability", color=color.purple, display=display.pane)')
pine_code.append('hline(0.5, "Neutral", color=color.gray, linestyle=hline.style_dashed)')

with open("strategy.pine", "w") as f:
    f.write("\n".join(pine_code))

print("Transpiled HGB proxy successfully to strategy.pine!")

import joblib
import numpy as np

# Load the trained model
model = joblib.load("universal_strategy.joblib")

feature_names = list(model.feature_names_in_)
baseline = float(model._baseline_prediction.item())
# Fixed attribute name based on scikit-learn 1.9.0 implementation
predictors = model._predictors  

pine_code = []
pine_code.append("//@version=5")
pine_code.append('strategy("Lossless GBDT Strategy (NSE)", overlay=true, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, process_orders_on_close=true)')
pine_code.append("")
pine_code.append("// --- 1. EXACT FEATURE RECONSTRUCTION ---")
pine_code.append("return_1d       = (close - close[1]) / close[1]")
pine_code.append("volatility_20d  = ta.stdev(return_1d, 20)")
pine_code.append("sma_20_ratio    = close / ta.sma(close, 20)")
pine_code.append("sma_50_ratio    = close / ta.sma(close, 50)")
pine_code.append("// Note: Use standard ta.rsi if trained on Wilder's RMA (TA-Lib default)")
pine_code.append("rsi_14          = ta.rsi(close, 14)")
pine_code.append("")

def render_node(nodes, idx):
    node = nodes[idx]
    if node["is_leaf"]:
        return f"{node['value']:.8f}"
    
    feat = feature_names[node["feature_idx"]]
    thresh = node["num_threshold"]
    left_repr = render_node(nodes, node["left"])
    right_repr = render_node(nodes, node["right"])
    
    # Binary split logic
    return f"({feat} <= {thresh:.8f} ? {left_repr} : {right_repr})"

# Group trees into chunks of 10 to prevent TradingView stack/compiler token limits
pine_code.append("// --- 2. COMPILED TREE ENSEMBLE ---")
chunk_size = 10
total_trees = len(predictors)
chunk_calls = []

for c_idx in range(0, total_trees, chunk_size):
    chunk_trees = predictors[c_idx:c_idx + chunk_size]
    func_name = f"calc_chunk_{c_idx // chunk_size}"
    chunk_calls.append(f"{func_name}()")
    
    pine_code.append(f"{func_name}() =>")
    pine_code.append("    float score = 0.0")
    for t_i, tree in enumerate(chunk_trees):
        expr = render_node(tree[0].nodes, 0)
        pine_code.append(f"    score += {expr}")
    pine_code.append("    score")
    pine_code.append("")

# Link function and inference
pine_code.append("// --- 3. INFERENCE & EXECUTION ---")
pine_code.append(f"float baseline = {baseline:.8f}")
pine_code.append(f"float raw_margin = baseline + {' + '.join(chunk_calls)}")
pine_code.append("float prob = 1.0 / (1.0 + math.exp(-raw_margin))")
pine_code.append("")
pine_code.append("input_threshold = input.float(0.55, 'Buy Probability Threshold', minval=0.50, maxval=0.95, step=0.01)")
pine_code.append("exit_threshold  = input.float(0.45, 'Exit/Sell Probability Threshold', minval=0.10, maxval=0.50, step=0.01)")
pine_code.append("")
pine_code.append("long_condition  = prob >= input_threshold and barstate.isconfirmed")
pine_code.append("exit_condition  = prob <= exit_threshold and barstate.isconfirmed")
pine_code.append("")
pine_code.append("if (long_condition)")
pine_code.append('    strategy.entry("Long", strategy.long)')
pine_code.append("if (exit_condition)")
pine_code.append('    strategy.close("Long")')
pine_code.append("")
pine_code.append('plot(prob, "Model Probability", color=color.purple, display=display.pane)')
pine_code.append('hline(0.5, "Neutral", color=color.gray, linestyle=hline.style_dashed)')

# Write to file
with open("strategy.pine", "w") as f:
    f.write("\n".join(pine_code))

print("Transpiled successfully to strategy.pine!")

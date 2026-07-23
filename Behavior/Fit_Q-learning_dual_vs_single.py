# -*- coding: utf-8 -*-
"""
Single- versus dual-learning-rate Q-learning model comparison.

Public-release version:
- No user names, drive letters, or machine-specific absolute paths are stored.
- Input and output locations are supplied through command-line arguments.
- The analysis logic and output structure are otherwise preserved.
"""

import argparse
import os
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import minimize
from scipy.special import expit


# ============================================================
# 路径设置
# ============================================================

def parse_arguments():
    """解析输入数据与输出目录，避免在公开代码中保留本地绝对路径。"""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Fit and compare single- and dual-learning-rate Q-learning models "
            "and export subject-level and trial-level results."
        )
    )
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=script_dir / "data" / "cbm_ready_data_cue24_block_by_id.mat",
        help=(
            "Path to the input MATLAB file. Default: "
            "<script_dir>/data/cbm_ready_data_cue24_block_by_id.mat"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "qlearning_model_comparison",
        help=(
            "Directory for all output files. Default: "
            "<script_dir>/outputs/qlearning_model_comparison"
        ),
    )
    return parser.parse_args()


# ============================================================
# 模型与数据设置
# ============================================================

GO_CODE = 1
NOGO_CODE = 2

# 若 action 是 0/1，这里默认 1 = Go, 0 = NoGo。
# 如果你的数据恰好相反，只需要修改这里。
ACTION_01_MAPPING = {1: GO_CODE, 0: NOGO_CODE}

# State key:
# "cue"        = Qs[cue_id][action]
# "block_cue"  = 每个 block 内 cue 重新初始化：
#                Qs[(block, cue_id)][action]
# "auto"       = 如果 cue_id 数量约等于 block × 2，则使用 cue；
#                否则使用 block_cue。
STATE_KEY_MODE = "auto"

ALPHA_BOUNDS = (1e-3, 0.999)
BETA_BOUNDS = (0.01, 50.0)

EPS = 1e-8
MAXITER = 5000


# ============================================================
# 工具函数
# ============================================================

def natural_sort_key(value):
    """使 Sub2 排在 Sub10 前面。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def safe_filename(value):
    """生成安全的文件名。"""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value))


def p_go_softmax(q_go, q_nogo, beta):
    """
    Two-action softmax:

        P(Go) = exp(beta * Qgo)
                / [exp(beta * Qgo) + exp(beta * Qnogo)]

              = sigmoid[beta * (Qgo - Qnogo)]
    """
    probability = expit(beta * (q_go - q_nogo))
    return float(np.clip(probability, EPS, 1.0 - EPS))


def make_state_key(cue, block, state_mode):
    """定义 Q table 的 state key。"""
    cue = int(cue)
    block = int(block)

    if state_mode == "cue":
        return cue

    if state_mode == "block_cue":
        return block, cue

    raise ValueError(f"Unknown state_mode: {state_mode}")


def infer_state_key_mode(dataframe, requested_mode="auto"):
    """
    自动判断 cue_id 是否已经是 block-specific。

    对于 cue24_block_by_id 形式的数据，通常：
        12 blocks × 2 cues = 24 cue_id

    因此使用 "cue"，保持每个 cue_id 独立。

    如果 cue_id 仅为原始 tactile pattern 编号，例如 1–8，
    则使用 "block_cue"，避免跨 block 继承 Q value。
    """
    if requested_mode in {"cue", "block_cue"}:
        return requested_mode

    if requested_mode != "auto":
        raise ValueError(
            "STATE_KEY_MODE must be 'auto', 'cue', or 'block_cue'."
        )

    number_of_blocks = int(dataframe["block"].nunique())
    number_of_cues = int(dataframe["cue_id"].nunique())

    if number_of_cues >= 2 * number_of_blocks:
        return "cue"

    return "block_cue"


def normalize_rewards(reward_raw):
    """
    将 reward 统一为 -0.5 / +0.5。

    - 如果原始 reward 是 0/1：转换为 -0.5/+0.5。
    - 如果已经是 -0.5/+0.5：保持不变。
    """
    rewards = pd.to_numeric(
        reward_raw,
        errors="coerce",
    ).astype(float)

    unique_values = set(
        np.round(
            rewards.dropna().unique(),
            6,
        )
    )

    if len(unique_values) == 0:
        raise ValueError("Reward column has no valid values.")

    if unique_values.issubset({0.0, 1.0}):
        return rewards - 0.5, "0/1_to_-0.5/+0.5"

    if unique_values.issubset({-0.5, 0.5}):
        return rewards, "already_-0.5/+0.5"

    # 宽松处理：如果 reward 在 [0, 1] 内，整体平移 -0.5。
    if (
        np.nanmin(rewards.values) >= 0.0
        and np.nanmax(rewards.values) <= 1.0
    ):
        return rewards - 0.5, "within_0/1_shifted_by_-0.5"

    # 如果已经在 [-0.5, 0.5] 内，则保持不变。
    if (
        np.nanmin(rewards.values) >= -0.5
        and np.nanmax(rewards.values) <= 0.5
    ):
        return rewards, "already_within_-0.5/+0.5"

    raise ValueError(
        f"Unexpected reward coding: {sorted(unique_values)}. "
        "Expected 0/1 or -0.5/+0.5."
    )


def normalize_actions(action_raw):
    """
    将 action 统一为：

        1 = Go
        2 = NoGo
    """
    actions = pd.to_numeric(
        action_raw,
        errors="coerce",
    )

    if actions.isna().any():
        raise ValueError(
            "Action column contains values that cannot be converted to numeric codes."
        )

    actions = actions.astype(int)
    unique_values = set(actions.unique())

    if unique_values.issubset({1, 2}):
        return actions, "already_1Go_2NoGo"

    if unique_values.issubset({0, 1}):
        mapped = actions.map(ACTION_01_MAPPING)

        if mapped.isna().any():
            raise ValueError(
                "Some 0/1 action codes could not be mapped to Go/NoGo."
            )

        return mapped.astype(int), "recode_0/1_to_1Go_2NoGo"

    raise ValueError(
        f"Unexpected action coding: {sorted(unique_values)}. "
        "Expected {1, 2} or {0, 1}."
    )


def prepare_subject_dataframe(subject, trials):
    """
    将 MATLAB 中每名参与者的矩阵转换为 DataFrame。

    预期前五列依次为：

        cue_id, reversal, action, reward, block
    """
    array = np.asarray(trials)

    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(
            f"{subject}: expected a 2D array with at least 5 columns, "
            f"got shape {array.shape}"
        )

    dataframe = pd.DataFrame(
        array[:, :5],
        columns=[
            "cue_id",
            "reversal",
            "action_raw",
            "reward_raw",
            "block",
        ],
    )

    required_columns = [
        "cue_id",
        "reversal",
        "action_raw",
        "reward_raw",
        "block",
    ]

    for column in required_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    number_before = len(dataframe)

    dataframe = dataframe.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    number_dropped = number_before - len(dataframe)

    if number_dropped > 0:
        print(
            f"Warning: {subject}: dropped {number_dropped} "
            "rows containing missing values."
        )

    dataframe["cue_id"] = dataframe["cue_id"].astype(int)
    dataframe["reversal"] = dataframe["reversal"].astype(int)
    dataframe["block"] = dataframe["block"].astype(int)

    dataframe["action"], action_coding = normalize_actions(
        dataframe["action_raw"]
    )
    dataframe["reward_model"], reward_coding = normalize_rewards(
        dataframe["reward_raw"]
    )

    dataframe.insert(0, "subject", subject)
    dataframe["global_trial"] = np.arange(
        1,
        len(dataframe) + 1,
    )

    # 实际的 block 内 trial index。
    dataframe["trial_in_block"] = (
        dataframe.groupby("block").cumcount() + 1
    )

    return dataframe, action_coding, reward_coding


def compute_information_criteria(nll, number_of_parameters, number_of_trials):
    """
    计算 AIC、BIC 和相对于随机二选一模型的 pseudo-R²。

        AIC = 2 × NLL + 2 × k
        BIC = 2 × NLL + k × log(n)

    随机模型假设每个 trial 上：

        P(choice) = 0.5
    """
    aic = 2.0 * nll + 2.0 * number_of_parameters
    bic = (
        2.0 * nll
        + float(number_of_parameters) * np.log(number_of_trials)
    )

    nll_random = number_of_trials * np.log(2.0)
    pseudo_r_squared = 1.0 - (nll / nll_random)

    return aic, bic, pseudo_r_squared


def bic_weights(bic_single, bic_dual):
    """计算基于 BIC 的近似 model weights。"""
    bics = np.array(
        [bic_single, bic_dual],
        dtype=float,
    )

    relative_likelihood = np.exp(
        -0.5 * (bics - np.min(bics))
    )
    weights = relative_likelihood / np.sum(relative_likelihood)

    return float(weights[0]), float(weights[1])


def near_bound(value, bounds, tolerance=1e-3):
    """检查参数估计是否接近优化边界。"""
    return bool(
        abs(value - bounds[0]) <= tolerance
        or abs(value - bounds[1]) <= tolerance
    )


# ============================================================
# 负对数似然函数
# ============================================================

def neg_log_likelihood_single(parameters, trials_all, state_mode):
    """
    Single-learning-rate Q-learning model。

    Parameters
    ----------
    parameters:
        [alpha, beta]

    trials_all columns:
        cue_id, action, reward_model, block
    """
    alpha, beta = parameters

    if not ALPHA_BOUNDS[0] <= alpha <= ALPHA_BOUNDS[1]:
        return np.inf

    if not BETA_BOUNDS[0] <= beta <= BETA_BOUNDS[1]:
        return np.inf

    q_values = defaultdict(
        lambda: {
            GO_CODE: 0.0,
            NOGO_CODE: 0.0,
        }
    )

    negative_log_likelihood = 0.0

    for cue, action, reward, block in trials_all:
        cue = int(cue)
        action = int(action)
        block = int(block)
        reward = float(reward)

        state = make_state_key(
            cue,
            block,
            state_mode,
        )

        q_go = q_values[state][GO_CODE]
        q_nogo = q_values[state][NOGO_CODE]

        probability_go = p_go_softmax(
            q_go,
            q_nogo,
            beta,
        )

        probability_chosen = (
            probability_go
            if action == GO_CODE
            else 1.0 - probability_go
        )

        probability_chosen = np.clip(
            probability_chosen,
            EPS,
            1.0 - EPS,
        )

        negative_log_likelihood -= np.log(
            probability_chosen
        )

        prediction_error = (
            reward - q_values[state][action]
        )

        q_values[state][action] += (
            alpha * prediction_error
        )

    return float(negative_log_likelihood)


def neg_log_likelihood_dual(parameters, trials_all, state_mode):
    """
    Dual-learning-rate Q-learning model。

    Parameters
    ----------
    parameters:
        [alpha_pos, alpha_neg, beta]

    根据 prediction error 的符号选择 learning rate：

        alpha_pos, if prediction_error >= 0
        alpha_neg, if prediction_error < 0
    """
    alpha_pos, alpha_neg, beta = parameters

    if not ALPHA_BOUNDS[0] <= alpha_pos <= ALPHA_BOUNDS[1]:
        return np.inf

    if not ALPHA_BOUNDS[0] <= alpha_neg <= ALPHA_BOUNDS[1]:
        return np.inf

    if not BETA_BOUNDS[0] <= beta <= BETA_BOUNDS[1]:
        return np.inf

    q_values = defaultdict(
        lambda: {
            GO_CODE: 0.0,
            NOGO_CODE: 0.0,
        }
    )

    negative_log_likelihood = 0.0

    for cue, action, reward, block in trials_all:
        cue = int(cue)
        action = int(action)
        block = int(block)
        reward = float(reward)

        state = make_state_key(
            cue,
            block,
            state_mode,
        )

        q_go = q_values[state][GO_CODE]
        q_nogo = q_values[state][NOGO_CODE]

        probability_go = p_go_softmax(
            q_go,
            q_nogo,
            beta,
        )

        probability_chosen = (
            probability_go
            if action == GO_CODE
            else 1.0 - probability_go
        )

        probability_chosen = np.clip(
            probability_chosen,
            EPS,
            1.0 - EPS,
        )

        negative_log_likelihood -= np.log(
            probability_chosen
        )

        prediction_error = (
            reward - q_values[state][action]
        )

        learning_rate = (
            alpha_pos
            if prediction_error >= 0
            else alpha_neg
        )

        q_values[state][action] += (
            learning_rate * prediction_error
        )

    return float(negative_log_likelihood)


# ============================================================
# Multi-start optimization
# ============================================================

def get_initial_points(model_name):
    """
    生成多组优化初始点。

    Multi-start optimization 可降低局部极小值影响。
    """
    alpha_grid = [0.05, 0.30, 0.70]
    beta_grid = [0.50, 3.00, 10.00, 30.00]

    if model_name == "single":
        return [
            [alpha, beta]
            for alpha, beta in product(
                alpha_grid,
                beta_grid,
            )
        ]

    if model_name == "dual":
        return [
            [alpha_pos, alpha_neg, beta]
            for alpha_pos, alpha_neg, beta in product(
                alpha_grid,
                alpha_grid,
                beta_grid,
            )
        ]

    raise ValueError(f"Unknown model_name: {model_name}")


def fit_model_multistart(model_name, trials_all, state_mode):
    """使用 L-BFGS-B 和多组初始点拟合指定模型。"""
    if model_name == "single":
        likelihood_function = neg_log_likelihood_single
        bounds = [
            ALPHA_BOUNDS,
            BETA_BOUNDS,
        ]

    elif model_name == "dual":
        likelihood_function = neg_log_likelihood_dual
        bounds = [
            ALPHA_BOUNDS,
            ALPHA_BOUNDS,
            BETA_BOUNDS,
        ]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    initial_points = get_initial_points(model_name)
    fit_details = []
    best_fit = None

    for start_index, initial_parameters in enumerate(
        initial_points,
        start=1,
    ):
        try:
            result = minimize(
                likelihood_function,
                x0=np.array(
                    initial_parameters,
                    dtype=float,
                ),
                args=(
                    trials_all,
                    state_mode,
                ),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": MAXITER,
                    "disp": False,
                },
            )

            objective_value = (
                float(result.fun)
                if np.isfinite(result.fun)
                else np.inf
            )

            fitted_parameters = (
                np.array(
                    result.x,
                    dtype=float,
                )
                if result.x is not None
                else np.full(
                    len(initial_parameters),
                    np.nan,
                )
            )

            detail = {
                "model": model_name,
                "start_idx": start_index,
                "init_params": ";".join(
                    f"{value:.6g}"
                    for value in initial_parameters
                ),
                "nll": objective_value,
                "success": bool(result.success),
                "message": str(result.message),
            }

            if model_name == "single":
                detail.update(
                    {
                        "alpha": fitted_parameters[0],
                        "beta": fitted_parameters[1],
                        "alpha_pos": np.nan,
                        "alpha_neg": np.nan,
                    }
                )

            else:
                detail.update(
                    {
                        "alpha": np.nan,
                        "alpha_pos": fitted_parameters[0],
                        "alpha_neg": fitted_parameters[1],
                        "beta": fitted_parameters[2],
                    }
                )

            fit_details.append(detail)

            if np.isfinite(objective_value):
                if (
                    best_fit is None
                    or objective_value < best_fit["nll"]
                ):
                    best_fit = {
                        "model": model_name,
                        "params": fitted_parameters,
                        "nll": objective_value,
                        "success": bool(result.success),
                        "message": str(result.message),
                        "start_idx": start_index,
                        "init_params": initial_parameters,
                    }

        except Exception as error:
            fit_details.append(
                {
                    "model": model_name,
                    "start_idx": start_index,
                    "init_params": ";".join(
                        f"{value:.6g}"
                        for value in initial_parameters
                    ),
                    "nll": np.inf,
                    "success": False,
                    "message": repr(error),
                    "alpha": np.nan,
                    "alpha_pos": np.nan,
                    "alpha_neg": np.nan,
                    "beta": np.nan,
                }
            )

    if best_fit is None:
        raise RuntimeError(
            f"No finite solution found for {model_name} model."
        )

    return best_fit, pd.DataFrame(fit_details)


# ============================================================
# 根据拟合参数重新生成 trial-wise Q / PE / policy
# ============================================================

def recompute_trialwise(
    dataframe,
    model_name,
    parameters,
    state_mode,
):
    """根据拟合参数重新计算逐试次模型变量。"""
    suffix = model_name

    q_values = defaultdict(
        lambda: {
            GO_CODE: 0.0,
            NOGO_CODE: 0.0,
        }
    )

    records = []

    if model_name == "single":
        alpha = float(parameters[0])
        beta = float(parameters[1])
        alpha_pos = np.nan
        alpha_neg = np.nan

    elif model_name == "dual":
        alpha_pos = float(parameters[0])
        alpha_neg = float(parameters[1])
        beta = float(parameters[2])
        alpha = np.nan

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    for _, row in dataframe.iterrows():
        cue = int(row["cue_id"])
        block = int(row["block"])
        action = int(row["action"])
        reward = float(row["reward_model"])

        state = make_state_key(
            cue,
            block,
            state_mode,
        )

        q_go_pre = q_values[state][GO_CODE]
        q_nogo_pre = q_values[state][NOGO_CODE]
        q_chosen_pre = q_values[state][action]

        probability_go = p_go_softmax(
            q_go_pre,
            q_nogo_pre,
            beta,
        )
        probability_nogo = 1.0 - probability_go

        probability_chosen = (
            probability_go
            if action == GO_CODE
            else probability_nogo
        )

        prediction_error = reward - q_chosen_pre

        if model_name == "single":
            alpha_used = alpha
        else:
            alpha_used = (
                alpha_pos
                if prediction_error >= 0
                else alpha_neg
            )

        q_values[state][action] += (
            alpha_used * prediction_error
        )

        q_go_post = q_values[state][GO_CODE]
        q_nogo_post = q_values[state][NOGO_CODE]

        records.append(
            {
                "global_trial": int(row["global_trial"]),
                "block": block,

                f"Q_go_pre_{suffix}": q_go_pre,
                f"Q_nogo_pre_{suffix}": q_nogo_pre,
                f"Q_chosen_pre_{suffix}": q_chosen_pre,

                f"p_go_{suffix}": probability_go,
                f"p_nogo_{suffix}": probability_nogo,
                f"p_chosen_{suffix}": probability_chosen,

                f"vPE_{suffix}": prediction_error,
                f"abs_vPE_{suffix}": abs(prediction_error),
                f"vPE_sign_{suffix}": (
                    "positive_or_zero"
                    if prediction_error >= 0
                    else "negative"
                ),

                f"alpha_used_{suffix}": alpha_used,
                f"beta_{suffix}": beta,

                f"Q_go_post_{suffix}": q_go_post,
                f"Q_nogo_post_{suffix}": q_nogo_post,
            }
        )

    output = pd.DataFrame(records)

    # ========================================================
    # Delta-policy definition
    # ========================================================
    #
    # delta_policy[t, t+1]
    #     = p_chosen[t] - p_chosen[t+1]
    #
    # 这里按照 global trial 顺序计算，而不是在每个 block 内计算。
    #
    # 因此：
    # - block 末尾 trial 不会被设为 NaN；
    # - 最后一个 global trial 没有 t+1；
    # - 最后一个 trial 的 next probability 设为当前值；
    # - 因而最后一个 trial 的 delta_policy = 0。
    # ========================================================

    output = output.sort_values(
        "global_trial"
    ).reset_index(drop=True)

    probability_column = f"p_chosen_{suffix}"
    next_probability_column = (
        f"p_chosen_next_{suffix}"
    )
    delta_policy_column = (
        f"delta_policy_{suffix}"
    )
    absolute_delta_policy_column = (
        f"abs_delta_policy_{suffix}"
    )

    # Global shift，不按 block 分组。
    output[next_probability_column] = (
        output[probability_column].shift(-1)
    )

    # 最后一个 trial：next = current，delta = 0。
    output[next_probability_column] = (
        output[next_probability_column].fillna(
            output[probability_column]
        )
    )

    output[delta_policy_column] = (
        output[probability_column]
        - output[next_probability_column]
    )
    output[absolute_delta_policy_column] = (
        output[delta_policy_column].abs()
    )

    # 额外保留 p_go 在 t 与 t+1 之间的变化。
    # 注意：delta_p_go 不等于 delta_policy。
    # 对于 NoGo trial，p_chosen = 1 - p_go。
    probability_go_column = f"p_go_{suffix}"
    next_probability_go_column = (
        f"p_go_next_{suffix}"
    )
    delta_probability_go_column = (
        f"delta_p_go_{suffix}"
    )

    output[next_probability_go_column] = (
        output[probability_go_column].shift(-1)
    )

    output[next_probability_go_column] = (
        output[next_probability_go_column].fillna(
            output[probability_go_column]
        )
    )

    output[delta_probability_go_column] = (
        output[probability_go_column]
        - output[next_probability_go_column]
    )

    return output


# ============================================================
# 主程序
# ============================================================

def main():
    args = parse_arguments()

    mat_path = args.mat_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    trialwise_dir = output_dir / "trialwise_combined"

    if not mat_path.is_file():
        raise FileNotFoundError(
            "Input MATLAB file was not found. "
            f"Provide a valid path with --mat-path: {mat_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    trialwise_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = sio.loadmat(str(mat_path))

    subject_keys = sorted(
        [
            key
            for key in data.keys()
            if not key.startswith("__")
        ],
        key=natural_sort_key,
    )

    print(
        f"Found {len(subject_keys)} "
        "subject-like variables in MAT file."
    )
    print(f"Input MATLAB file: {mat_path}")
    print(f"Output directory: {output_dir}")
    print(
        "Dpolicy definition: "
        "p_chosen[t] - p_chosen[t+1]"
    )
    print(
        "Dpolicy shift mode: GLOBAL consecutive trials, "
        "no block-end NaNs."
    )
    print(
        "Final global trial: delta_policy = 0."
    )

    summary_rows = []
    all_trialwise = []
    all_optimization_details = []

    for subject in subject_keys:
        print("\n" + "=" * 80)
        print(f"Fitting subject: {subject}")

        try:
            (
                subject_dataframe,
                action_coding,
                reward_coding,
            ) = prepare_subject_dataframe(
                subject,
                data[subject],
            )

        except Exception as error:
            print(
                f"Error: {subject}: failed during "
                f"data preparation: {error}"
            )
            continue

        if subject_dataframe["action"].nunique() < 2:
            print(
                f"Warning: {subject}: only one unique "
                "action after recoding; skipped."
            )
            continue

        try:
            state_mode = infer_state_key_mode(
                subject_dataframe,
                STATE_KEY_MODE,
            )

        except Exception as error:
            print(
                f"Error: {subject}: failed to infer "
                f"state mode: {error}"
            )
            continue

        number_of_trials = len(subject_dataframe)

        trials_all = subject_dataframe[
            [
                "cue_id",
                "action",
                "reward_model",
                "block",
            ]
        ].to_numpy(dtype=float)

        print(
            f"{subject}: "
            f"n_trials={number_of_trials}, "
            f"action_coding={action_coding}, "
            f"reward_coding={reward_coding}, "
            f"state_mode={state_mode}"
        )

        try:
            fit_single, details_single = (
                fit_model_multistart(
                    "single",
                    trials_all,
                    state_mode,
                )
            )

            fit_dual, details_dual = (
                fit_model_multistart(
                    "dual",
                    trials_all,
                    state_mode,
                )
            )

        except Exception as error:
            print(
                f"Error: {subject}: fitting failed: {error}"
            )
            continue

        details_single.insert(
            0,
            "subject",
            subject,
        )
        details_dual.insert(
            0,
            "subject",
            subject,
        )

        all_optimization_details.append(
            details_single
        )
        all_optimization_details.append(
            details_dual
        )

        # ====================================================
        # Fit metrics
        # ====================================================

        nll_single = fit_single["nll"]
        nll_dual = fit_dual["nll"]

        alpha_single, beta_single = (
            fit_single["params"]
        )
        alpha_pos, alpha_neg, beta_dual = (
            fit_dual["params"]
        )

        (
            aic_single,
            bic_single,
            pseudo_r_squared_single,
        ) = compute_information_criteria(
            nll_single,
            number_of_parameters=2,
            number_of_trials=number_of_trials,
        )

        (
            aic_dual,
            bic_dual,
            pseudo_r_squared_dual,
        ) = compute_information_criteria(
            nll_dual,
            number_of_parameters=3,
            number_of_trials=number_of_trials,
        )

        delta_nll = nll_dual - nll_single
        delta_aic = aic_dual - aic_single
        delta_bic = bic_dual - bic_single

        bic_weight_single, bic_weight_dual = (
            bic_weights(
                bic_single,
                bic_dual,
            )
        )

        winner_by_aic = (
            "dual"
            if aic_dual < aic_single
            else "single"
        )
        winner_by_bic = (
            "dual"
            if bic_dual < bic_single
            else "single"
        )

        summary_rows.append(
            {
                "subject": subject,
                "n_trials": number_of_trials,
                "action_coding": action_coding,
                "reward_coding": reward_coding,
                "state_key_mode": state_mode,

                "nll_single": nll_single,
                "aic_single": aic_single,
                "bic_single": bic_single,
                "pseudoR2_vs_random_single": (
                    pseudo_r_squared_single
                ),
                "alpha_single": alpha_single,
                "beta_single": beta_single,
                "single_success": fit_single["success"],
                "single_message": fit_single["message"],
                "single_best_start_idx": (
                    fit_single["start_idx"]
                ),

                "nll_dual": nll_dual,
                "aic_dual": aic_dual,
                "bic_dual": bic_dual,
                "pseudoR2_vs_random_dual": (
                    pseudo_r_squared_dual
                ),
                "alpha_pos": alpha_pos,
                "alpha_neg": alpha_neg,
                "alpha_neg_minus_pos": (
                    alpha_neg - alpha_pos
                ),
                "beta_dual": beta_dual,
                "dual_success": fit_dual["success"],
                "dual_message": fit_dual["message"],
                "dual_best_start_idx": (
                    fit_dual["start_idx"]
                ),

                "delta_nll_dual_minus_single": (
                    delta_nll
                ),
                "delta_aic_dual_minus_single": (
                    delta_aic
                ),
                "delta_bic_dual_minus_single": (
                    delta_bic
                ),

                "bic_weight_single": bic_weight_single,
                "bic_weight_dual": bic_weight_dual,
                "winner_by_aic": winner_by_aic,
                "winner_by_bic": winner_by_bic,

                "alpha_single_at_bound": near_bound(
                    alpha_single,
                    ALPHA_BOUNDS,
                ),
                "beta_single_at_bound": near_bound(
                    beta_single,
                    BETA_BOUNDS,
                ),
                "alpha_pos_at_bound": near_bound(
                    alpha_pos,
                    ALPHA_BOUNDS,
                ),
                "alpha_neg_at_bound": near_bound(
                    alpha_neg,
                    ALPHA_BOUNDS,
                ),
                "beta_dual_at_bound": near_bound(
                    beta_dual,
                    BETA_BOUNDS,
                ),
            }
        )

        print(
            f"{subject}: "
            f"NLL single={nll_single:.3f}, "
            f"dual={nll_dual:.3f}; "
            f"BIC single={bic_single:.3f}, "
            f"dual={bic_dual:.3f}; "
            f"delta BIC dual-single={delta_bic:.3f}; "
            f"winner={winner_by_bic}"
        )

        # ====================================================
        # Trial-wise recomputation
        # ====================================================

        trial_single = recompute_trialwise(
            subject_dataframe,
            "single",
            fit_single["params"],
            state_mode,
        )

        trial_dual = recompute_trialwise(
            subject_dataframe,
            "dual",
            fit_dual["params"],
            state_mode,
        )

        base_columns = [
            "subject",
            "global_trial",
            "block",
            "trial_in_block",
            "cue_id",
            "reversal",
            "action_raw",
            "action",
            "reward_raw",
            "reward_model",
        ]

        combined_dataframe = (
            subject_dataframe[base_columns].copy()
        )

        combined_dataframe["state_key_mode"] = (
            state_mode
        )
        combined_dataframe["winner_by_bic"] = (
            winner_by_bic
        )

        combined_dataframe = combined_dataframe.join(
            trial_single.drop(
                columns=[
                    "global_trial",
                    "block",
                ]
            )
        )

        combined_dataframe = combined_dataframe.join(
            trial_dual.drop(
                columns=[
                    "global_trial",
                    "block",
                ]
            )
        )

        # 为后续分析提供按 BIC 胜出模型生成的变量。
        if winner_by_bic == "dual":
            combined_dataframe["vPE_winner_BIC"] = (
                combined_dataframe["vPE_dual"]
            )
            combined_dataframe[
                "abs_vPE_winner_BIC"
            ] = combined_dataframe["abs_vPE_dual"]

            combined_dataframe["p_go_winner_BIC"] = (
                combined_dataframe["p_go_dual"]
            )
            combined_dataframe[
                "p_go_next_winner_BIC"
            ] = combined_dataframe["p_go_next_dual"]
            combined_dataframe[
                "delta_p_go_winner_BIC"
            ] = combined_dataframe["delta_p_go_dual"]

            combined_dataframe[
                "p_chosen_winner_BIC"
            ] = combined_dataframe["p_chosen_dual"]
            combined_dataframe[
                "p_chosen_next_winner_BIC"
            ] = combined_dataframe[
                "p_chosen_next_dual"
            ]

            combined_dataframe[
                "delta_policy_winner_BIC"
            ] = combined_dataframe["delta_policy_dual"]
            combined_dataframe[
                "abs_delta_policy_winner_BIC"
            ] = combined_dataframe[
                "abs_delta_policy_dual"
            ]

        else:
            combined_dataframe["vPE_winner_BIC"] = (
                combined_dataframe["vPE_single"]
            )
            combined_dataframe[
                "abs_vPE_winner_BIC"
            ] = combined_dataframe["abs_vPE_single"]

            combined_dataframe["p_go_winner_BIC"] = (
                combined_dataframe["p_go_single"]
            )
            combined_dataframe[
                "p_go_next_winner_BIC"
            ] = combined_dataframe["p_go_next_single"]
            combined_dataframe[
                "delta_p_go_winner_BIC"
            ] = combined_dataframe["delta_p_go_single"]

            combined_dataframe[
                "p_chosen_winner_BIC"
            ] = combined_dataframe["p_chosen_single"]
            combined_dataframe[
                "p_chosen_next_winner_BIC"
            ] = combined_dataframe[
                "p_chosen_next_single"
            ]

            combined_dataframe[
                "delta_policy_winner_BIC"
            ] = combined_dataframe["delta_policy_single"]
            combined_dataframe[
                "abs_delta_policy_winner_BIC"
            ] = combined_dataframe[
                "abs_delta_policy_single"
            ]

        subject_output_path = (
            trialwise_dir
            / (
                "Qlearning_single_dual_trialwise_"
                f"{safe_filename(subject)}.csv"
            )
        )

        combined_dataframe.to_csv(
            subject_output_path,
            index=False,
            encoding="utf-8-sig",
        )

        all_trialwise.append(
            combined_dataframe
        )

        print(
            "Saved trial-wise output: "
            f"{subject_output_path}"
        )

    # ========================================================
    # 保存参与者层面与组层面结果
    # ========================================================

    if len(summary_rows) == 0:
        print(
            "\nNo subjects were successfully fitted."
        )
        return

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        output_dir
        / "Qlearning_subjectwise_single_dual_model_comparison.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    if len(all_trialwise) > 0:
        all_trialwise_dataframe = pd.concat(
            all_trialwise,
            axis=0,
            ignore_index=True,
        )

        all_trialwise_path = (
            output_dir
            / "Qlearning_all_subjects_trialwise_single_dual.csv"
        )

        all_trialwise_dataframe.to_csv(
            all_trialwise_path,
            index=False,
            encoding="utf-8-sig",
        )

    if len(all_optimization_details) > 0:
        optimization_dataframe = pd.concat(
            all_optimization_details,
            axis=0,
            ignore_index=True,
        )

        optimization_path = (
            output_dir
            / "Qlearning_multistart_optimization_details.csv"
        )

        optimization_dataframe.to_csv(
            optimization_path,
            index=False,
            encoding="utf-8-sig",
        )

    number_of_subjects = len(
        summary_dataframe
    )
    number_of_trials_total = int(
        summary_dataframe["n_trials"].sum()
    )

    summed_nll_single = float(
        summary_dataframe["nll_single"].sum()
    )
    summed_nll_dual = float(
        summary_dataframe["nll_dual"].sum()
    )

    summed_aic_single = float(
        summary_dataframe["aic_single"].sum()
    )
    summed_aic_dual = float(
        summary_dataframe["aic_dual"].sum()
    )

    summed_bic_single = float(
        summary_dataframe["bic_single"].sum()
    )
    summed_bic_dual = float(
        summary_dataframe["bic_dual"].sum()
    )

    group_row = {
        "n_subjects": number_of_subjects,
        "n_trials_total": number_of_trials_total,

        "summed_nll_single": summed_nll_single,
        "summed_nll_dual": summed_nll_dual,
        "delta_summed_nll_dual_minus_single": (
            summed_nll_dual - summed_nll_single
        ),

        "summed_aic_single": summed_aic_single,
        "summed_aic_dual": summed_aic_dual,
        "delta_summed_aic_dual_minus_single": (
            summed_aic_dual - summed_aic_single
        ),
        "group_winner_by_summed_aic": (
            "dual"
            if summed_aic_dual < summed_aic_single
            else "single"
        ),

        "summed_bic_single": summed_bic_single,
        "summed_bic_dual": summed_bic_dual,
        "delta_summed_bic_dual_minus_single": (
            summed_bic_dual - summed_bic_single
        ),
        "group_winner_by_summed_bic": (
            "dual"
            if summed_bic_dual < summed_bic_single
            else "single"
        ),

        "mean_delta_bic_dual_minus_single": float(
            summary_dataframe[
                "delta_bic_dual_minus_single"
            ].mean()
        ),
        "median_delta_bic_dual_minus_single": float(
            summary_dataframe[
                "delta_bic_dual_minus_single"
            ].median()
        ),

        "n_subjects_dual_wins_bic": int(
            (
                summary_dataframe["winner_by_bic"]
                == "dual"
            ).sum()
        ),
        "n_subjects_single_wins_bic": int(
            (
                summary_dataframe["winner_by_bic"]
                == "single"
            ).sum()
        ),

        "mean_alpha_single": float(
            summary_dataframe["alpha_single"].mean()
        ),
        "mean_alpha_pos": float(
            summary_dataframe["alpha_pos"].mean()
        ),
        "mean_alpha_neg": float(
            summary_dataframe["alpha_neg"].mean()
        ),
        "mean_alpha_neg_minus_pos": float(
            summary_dataframe[
                "alpha_neg_minus_pos"
            ].mean()
        ),

        "mean_beta_single": float(
            summary_dataframe["beta_single"].mean()
        ),
        "mean_beta_dual": float(
            summary_dataframe["beta_dual"].mean()
        ),

        "n_beta_single_at_bound": int(
            summary_dataframe[
                "beta_single_at_bound"
            ].sum()
        ),
        "n_beta_dual_at_bound": int(
            summary_dataframe[
                "beta_dual_at_bound"
            ].sum()
        ),
    }

    group_dataframe = pd.DataFrame(
        [group_row]
    )

    group_path = (
        output_dir
        / "Qlearning_group_single_dual_model_comparison.csv"
    )

    group_dataframe.to_csv(
        group_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print(
        "Finished all model fitting and comparison."
    )
    print(
        "Dpolicy definition: "
        "p_chosen[t] - p_chosen[t+1]"
    )
    print(
        "Dpolicy shift mode: GLOBAL consecutive trials, "
        "with no block-end missing values."
    )
    print(
        "Final global trial: delta_policy = 0."
    )
    print(
        "Positive Dpolicy = divergence from the current policy "
        "/ behavioral change becomes more likely."
    )
    print(
        "Negative Dpolicy = reinforcement of the current policy "
        "/ behavioral change becomes less likely."
    )
    print(
        f"Subject-wise summary: {summary_path}"
    )
    print(
        f"Group summary:        {group_path}"
    )
    print(
        f"Trial-wise folder:    {trialwise_dir}"
    )

    print("\nGroup-level model comparison:")
    print(
        f"Summed BIC single = "
        f"{summed_bic_single:.3f}"
    )
    print(
        f"Summed BIC dual   = "
        f"{summed_bic_dual:.3f}"
    )
    print(
        "Delta summed BIC dual - single = "
        f"{summed_bic_dual - summed_bic_single:.3f}"
    )
    print(
        "Group winner by summed BIC = "
        f"{group_row['group_winner_by_summed_bic']}"
    )
    print(
        "BIC subject winners: "
        f"dual={group_row['n_subjects_dual_wins_bic']}, "
        f"single={group_row['n_subjects_single_wins_bic']}"
    )


if __name__ == "__main__":
    main()
import streamlit as st
import math
import pandas as pd
from fractions import Fraction
from typing import List, Dict, Any, Tuple
import numpy as np

# --- Константы ---
EPSILON = 1e-6
Z_MIN = 17
K_PLANET = 3
f_PLANET = 1

# --- Параметры ---
TARGET_U = 400.0
MODULE_1 = 3.0
MODULE_2 = 5.0
I12_LIMITS = (2.0, 6.0)
I3H_LIMITS = (-60.0, -10.0)
I56_LIMITS = (2.0, 6.0)

# --- Проверка условий ---

def check_assembly_condition(z2_prime: int, z4: int, k_planet: int) -> bool:
    if k_planet == 0:
        return False
    return (z2_prime + z4) % k_planet == 0


def check_proximity_condition(z2_prime: int, z3: int, k_planet: int, f: int = 1) -> bool:
    if k_planet <= 1:
        return True
    lhs = math.sin(math.radians(180.0 / k_planet))
    denominator = z3 + z2_prime
    if denominator <= 0:
        return False
    rhs = (z2_prime + 2 * f) / denominator
    return lhs > rhs


# --- Вспомогательные функции ---

def find_first_integer_z(i_ratio: float, start_z1: int = Z_MIN, max_iter: int = 200) -> Tuple[int, int]:
    for z1 in range(start_z1, start_z1 + max_iter):
        z2_float = i_ratio * z1
        if abs(z2_float - round(z2_float)) < EPSILON:
            z2_integer = int(round(z2_float))
            if z2_integer >= Z_MIN:
                return z1, z2_integer
    return 0, 0


def find_combinations(target: float, i12_limits, i3h_limits, i56_limits) -> pd.DataFrame:
    combinations = []
    i12_start_int = int(i12_limits[0] * 10)
    i12_end_int = int(i12_limits[1] * 10)
    i56_start_int = int(i56_limits[0] * 10)
    i56_end_int = int(i56_limits[1] * 10)
    
    for i12_int in range(i12_start_int, i12_end_int + 1):
        i12 = i12_int / 10.0
        for i56_int in range(i56_start_int, i56_end_int + 1):
            i56 = i56_int / 10.0
            denominator = i12 * i56
            if abs(denominator) < EPSILON:
                continue
            i3_H_float = -target / denominator
            i3_H_rounded = round(i3_H_float, 1)
            if i3h_limits[0] - EPSILON <= i3_H_rounded <= i3h_limits[1] + EPSILON:
                product = i12 * i3_H_rounded * i56
                if abs(abs(product) - target) < EPSILON:
                    combinations.append({
                        "i12": i12,
                        "i3_H": i3_H_rounded,
                        "i56": i56,
                        "Product": round(abs(product), 2)
                    })
    df = pd.DataFrame(combinations)
    if not df.empty:
        df.drop_duplicates(subset=['i12', 'i3_H', 'i56'], inplace=True)
        df['|i3_H|'] = df['i3_H'].abs()
        df.sort_values(by=['|i3_H|', 'i12', 'i56'], inplace=True, ascending=[False, True, True])
        df.reset_index(drop=True, inplace=True)
    return df.drop(columns=['|i3_H|'])


def find_subcombinations(i3_H: float) -> List[Dict[str, Any]]:
    global K_PLANET, f_PLANET
    target_factor = 1.0 - i3_H
    max_c = 100
    subcombinations = []

    for c3 in range(1, max_c + 1):
        min_c2_ = math.ceil(c3 / 6)
        for c2_ in range(max(1, min_c2_), max_c + 1): 
            for c4 in range(1, max_c + 1):
                max_c3_ = c4 // 6
                if max_c3_ < 1:
                    continue
                for c3_ in range(1, max_c3_ + 1):
                    product_l = c3 * c4
                    product_r = target_factor * c2_ * c3_
                    if abs(product_l - product_r) < EPSILON:
                        c_list = [c2_, c3, c3_, c4]
                        min_c = min(c_list)
                        gamma_min = math.ceil(Z_MIN / min_c)
                        z2_ = c2_ * gamma_min
                        z3 = c3 * gamma_min
                        z3_ = c3_ * gamma_min
                        z4 = c4 * gamma_min
                        if not check_assembly_condition(z2_, z4, K_PLANET):
                            continue
                        if not check_proximity_condition(z2_, z3, K_PLANET, f_PLANET):
                            continue
                        q = c3 + c2_
                        p = c4 + c3_
                        g1 = MODULE_1 * (z2_ + z3)
                        g2 = MODULE_1 * (z3_ + z4)
                        g_planet = g1 + g2
                        subcombinations.append({
                            "c2_": c2_, "c3": c3, "c3_": c3_, "c4": c4,
                            "q": q, "p": p, "pDevq": round(p / q, 3),
                            "pPlusq": p + q, "gamma": gamma_min,
                            "Z2'": z2_, "Z3": z3, "Z3'": z3_, "Z4": z4,
                            "G_Planet_Sum": round(g_planet, 2)
                        })
    subcombinations.sort(key=lambda param: (param['pPlusq'], abs(param['pDevq'] - 1)))
    return subcombinations


def calculate_final_gabarits(df_best: pd.DataFrame, i12: float, i56: float) -> pd.DataFrame:
    results = []
    z1, z2 = find_first_integer_z(i12, Z_MIN)
    g3 = MODULE_1 * (z1 + z2)
    z5, z6 = find_first_integer_z(i56, Z_MIN)
    g4 = MODULE_2 * (z5 + z6)

    for index, row in df_best.iterrows():
        z2_planet = row["Z2'"]
        z3_planet = row["Z3"]
        g_total_sum = round(g3 + row["G_Planet_Sum"] + g4, 2)
        lhs = math.sin(math.radians(180.0 / K_PLANET))
        rhs = (z2_planet + 2 * f_PLANET) / (z3_planet + z2_planet) if (z3_planet + z2_planet) > 0 else 999
        res = {
            "i12": i12, "i3_H": row["i3_H"], "i56": i56,
            "Z1": z1, "Z2": z2, "Z2'": z2_planet, "Z3": z3_planet,
            "Z3'": row["Z3'"], "Z4": row["Z4"], "Z5": z5, "Z6": z6,
            "G_TOTAL (мм)": g_total_sum,
            "Проверка сборки": (row["Z2'"] + row["Z4"]) % K_PLANET,
            "Проверка соседства": f"{lhs:.4f} > {rhs:.4f} ({'ДА' if lhs > rhs else 'НЕТ'})"
        }
        results.append(res)
    df_final = pd.DataFrame(results)
    df_final.sort_values(by="G_TOTAL (мм)", ascending=True, inplace=True)
    return df_final


# --- Streamlit интерфейс ---
def app():
    st.set_page_config(layout="wide", page_title="Синтез сложного зубчатого механизма")
    st.title("⚙️ Синтез сложного зубчатого механизма")
    st.markdown("Это приложение выполняет расчёт числа зубьев и габаритов по заданному общему передаточному отношению.")

    st.sidebar.header("Параметры")
    st.sidebar.markdown("**Этап 1: Выбор передаточных чисел**")
    global TARGET_U, Z_MIN, K_PLANET
    target_u_input = st.sidebar.number_input("Целевое общее передаточное отношение (U₁₋₆)", value=TARGET_U, min_value=1.0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Этап 2: Планетарный механизм**")
    Z_MIN = st.sidebar.number_input("Минимальное число зубьев (Zmin)", value=Z_MIN, min_value=1, step=1)
    K_PLANET = st.sidebar.number_input("Количество сателлитов (K)", value=3, min_value=1, step=1)
    st.sidebar.info(f"Проверка сборки: (Z₂' + Z₄) mod {K_PLANET} = 0")
    st.sidebar.info(f"Проверка соседства: sin(180/{K_PLANET}) > (Z₂'+ 2f)/(Z₃ + Z₂')")

    st.header("Этап 1: Расчёт передаточных чисел")
    st.info(f"Условие: i₁₂ × i₂'₋H × i₅₆ = -{target_u_input}")

    if st.button("Найти комбинации (Таблица 2)"):
        df_combinations = find_combinations(target_u_input, I12_LIMITS, I3H_LIMITS, I56_LIMITS)
        st.session_state['df_combinations'] = df_combinations

    if 'df_combinations' in st.session_state and not st.session_state['df_combinations'].empty:
        st.subheader("Таблица 2: Комбинации передаточных чисел")
        st.dataframe(st.session_state['df_combinations'].style.format({'i12': '{:.1f}', 'i3_H': '{:.1f}', 'i56': '{:.1f}', 'Product': '{:.1f}'}))

if __name__ == "__main__":
    app()

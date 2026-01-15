import pandas as pd 
import numpy as np

#===========
# CONTENTS #
#===========
# hms_to_s
# s_to_hms
# subtract_hms_times
# adjust_volume
#

def hms_to_s(hms_time: str) -> int:
    h, m, s = [int(x) for x in hms_time.split(":")]
    total_s = h * 3600 + m * 60 + s

    return total_s

def s_to_hms(total_s: int) -> str:
    h = total_s // 3600
    m = (total_s - h*3600) // 60
    s = total_s - h*3600 - m*60

    h_digits = len(str(h))
    m_digits = len(str(m))
    s_digits = len(str(s))

    h_len2 = "0" + str(h) if h_digits == 1 else str(h)
    m_len2 = "0" + str(m) if m_digits == 1 else str(m)
    s_len2 = "0" + str(s) if s_digits == 1 else str(s)
    
    hms_str = f"{h_len2}:{m_len2}:{s_len2}"
    return hms_str

def subtract_hms_times(minuend: str, subtrahend: str) -> str: # 8-5=3; 8 -> minuend, 5 -> subtrahend
    # Return NaN if a value is missing
    if pd.isna(minuend) or pd.isna(subtrahend):
        return np.nan

    # Convert to s
    minuend_sec = hms_to_s(minuend)
    subtrahend_sec = hms_to_s(subtrahend)

    # Subtract s
    diff_sec = minuend_sec - subtrahend_sec

    # Convert to hms
    diff_hms = s_to_hms(diff_sec)

    return diff_hms

def adjust_volume(vol_ul: float, delay_t_hms: str) -> float:
    if pd.isna(delay_t_hms):
        return np.nan 
    if str(delay_t_hms).lower() == "nan":
        return np.nan 

    delay_t_sec = hms_to_s(delay_t_hms)

    if vol_ul < 250:
        if delay_t_sec >= 6300: # 01:45:00
            return vol_ul * 1.25
        elif delay_t_sec >= 5400: # 01:30:00
            return vol_ul * 1.2
        elif delay_t_sec >= 4500: # 01:15:00
            return vol_ul * 1.1
        elif delay_t_sec >= 3600: # 01:00:00
            return vol_ul * 1.08
        else: 
            return vol_ul
        
    if vol_ul >= 250:
        if delay_t_sec >= 6300: # 01:45:00
            return vol_ul * 1.15
        elif delay_t_sec >= 5400: # 01:30:00
            return vol_ul * 1.1
        elif delay_t_sec >= 4500: # 01:15:00
            return vol_ul * 1.07
        elif delay_t_sec >= 3600: # 01:00:00
            return vol_ul * 1.05
        else: 
            return vol_ul
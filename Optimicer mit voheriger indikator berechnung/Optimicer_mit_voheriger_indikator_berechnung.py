import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, date
import pyopencl as cl
import ccxt
import time
import csv
import os
import psutil
from itertools import product

# OpenCL Setup
platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0] if platform.get_devices(cl.device_type.GPU) else platform.get_devices(cl.device_type.CPU)[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Speicherüberwachung
def get_available_memory():
    mem = psutil.virtual_memory()
    return mem.available / 1024**3

def adjust_chunk_size(base_size, min_size=1000):
    available_gb = get_available_memory()
    gpu_mem = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) / 1024**3
    safe_size = min(base_size, int((min(available_gb * 0.8, gpu_mem * 0.7)) * 1024**3 / (5 * np.float32().nbytes)))
    return max(min_size, safe_size)

# Daten von Binance abrufen
def fetch_data(start_date, end_date, gui_callback):
    exchange = ccxt.binance()
    timeframe = '15m'
    symbol = 'ETH/USDT'
    # Konvertiere datetime.date zu datetime.datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.min.time())
    since = int(start_datetime.timestamp() * 1000)
    end_ms = int(end_datetime.timestamp() * 1000)
    ohlcv = []
    
    gui_callback("Lade Daten von Binance...\n")
    while since < end_ms:
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not data:
            break
        ohlcv.extend(data)
        since = data[-1][0] + 1
        time.sleep(0.5)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close']].astype(np.float32)
    df.to_pickle('ohlc_data.pkl')  # OHLC-Daten zwischenspeichern
    gui_callback(f"Daten geladen: {len(df)} Kerzen\n")
    return df

# OpenCL Kernel für Indikatoren und Strategie
kernel_code = """
__kernel void calculate_indicators(
    __global const float *close, __global const float *high, __global const float *low,
    __global float *rsi, __global float *ema_short, __global float *ema_long,
    __global float *atr, __global float *momentum,
    __global float *avg_gain, __global float *avg_loss,
    __global float *highest_high, __global float *lowest_low,
    int n, int rsi_length, int ema_short_period, int ema_long_period, int lookback) {
    int i = get_global_id(0);
    if (i >= n) return;

    // Momentum
    momentum[i] = i > 0 ? close[i] - close[i-1] : 0.0f;

    // EMA
    float ema_short_alpha = 2.0f / (ema_short_period + 1);
    float ema_long_alpha = 2.0f / (ema_long_period + 1);
    if (i == 0) {
        ema_short[i] = close[i];
        ema_long[i] = close[i];
    } else {
        ema_short[i] = ema_short_alpha * close[i] + (1.0f - ema_short_alpha) * ema_short[i-1];
        ema_long[i] = ema_long_alpha * close[i] + (1.0f - ema_long_alpha) * ema_long[i-1];
    }

    // RSI
    if (i >= rsi_length) {
        if (i == rsi_length) {
            float gain_sum = 0.0f, loss_sum = 0.0f;
            for (int j = 1; j <= rsi_length; j++) {
                float delta = close[i-j+1] - close[i-j];
                gain_sum += max(delta, 0.0f);
                loss_sum += max(-delta, 0.0f);
            }
            avg_gain[i] = gain_sum / rsi_length;
            avg_loss[i] = loss_sum / rsi_length;
        } else {
            float delta = close[i] - close[i-1];
            avg_gain[i] = (avg_gain[i-1] * (rsi_length - 1) + max(delta, 0.0f)) / rsi_length;
            avg_loss[i] = (avg_loss[i-1] * (rsi_length - 1) + max(-delta, 0.0f)) / rsi_length;
        }
        float rs = avg_loss[i] == 0 ? (avg_gain[i] == 0 ? 0.0f : 100.0f) : avg_gain[i] / avg_loss[i];
        rsi[i] = 100.0f - (100.0f / (1.0f + rs));
    } else {
        rsi[i] = 0.0f;
    }

    // ATR
    if (i >= 14) {
        float tr = fmax(fmax(high[i] - low[i], fabs(high[i] - close[i-1])), fabs(low[i] - close[i-1]));
        if (i == 14) {
            float sum_tr = 0.0f;
            for (int j = 0; j < 14; j++) {
                sum_tr += fmax(fmax(high[i-j] - low[i-j], fabs(high[i-j] - close[i-j-1])), fabs(low[i-j] - close[i-j-1]));
            }
            atr[i] = sum_tr / 14.0f;
        } else {
            atr[i] = (atr[i-1] * 13.0f + tr) / 14.0f;
        }
    } else {
        atr[i] = 0.0f;
    }

    // Highest High und Lowest Low
    if (i >= lookback) {
        float max_high = high[i];
        float min_low = low[i];
        for (int j = 1; j < lookback; j++) {
            max_high = fmax(max_high, high[i-j]);
            min_low = fmin(min_low, low[i-j]);
        }
        highest_high[i] = max_high;
        lowest_low[i] = min_low;
    } else {
        highest_high[i] = high[i];
        lowest_low[i] = low[i];
    }
}

__kernel void simulate_strategy(
    __global const float *close, __global const float *high, __global const float *low,
    __global const float *rsi, __global const float *ema_short, __global const float *ema_long,
    __global const float *atr, __global const float *momentum,
    __global const float *highest_high, __global const float *lowest_low,
    __global float *results, int n, int num_combinations,
    __global const float *params, float max_drawdown_limit) {
    int idx = get_global_id(0);
    if (idx >= num_combinations) return;

    int p_idx = idx * 13;
    float rsi_os = params[p_idx], rsi_ob = params[p_idx+1];
    float ema_short_period = params[p_idx+2], ema_long_period = params[p_idx+3];
    float sl_atr = params[p_idx+4], sl_percent = params[p_idx+5], sl_points = params[p_idx+6];
    float tp_atr = params[p_idx+7], tp_percent = params[p_idx+8], tp_points = params[p_idx+9];
    float risk_pct = params[p_idx+10];
    float entry_thresh = params[p_idx+11];
    int lookback = (int)params[p_idx+12];

    float equity = 10000.0f, position = 0.0f, entry_price = 0.0f, stop_loss = 0.0f, take_profit = 0.0f;
    float max_drawdown = 0.0f, peak = equity;
    int num_trades = 0, wins = 0;

    for (int i = max(14, lookback); i < n-1; i++) {
        float buy_score = (momentum[i] > 0 ? 1.0f : 0.0f) +
                          (rsi[i] < rsi_os ? 1.0f : 0.0f) +
                          (ema_short[i] > ema_long[i] ? 1.0f : 0.0f);
        float sell_score = (momentum[i] < 0 ? 1.0f : 0.0f) +
                           (rsi[i] > rsi_ob ? 1.0f : 0.0f) +
                           (ema_short[i] < ema_long[i] ? 1.0f : 0.0f);
        
        bool buy_breakout = close[i] > highest_high[i];
        bool sell_breakout = close[i] < lowest_low[i];

        if (position == 0 && (buy_score >= entry_thresh || buy_breakout)) {
            float stop_distance = sl_atr * atr[i];
            float position_size = (equity * risk_pct / 100.0f) / stop_distance;
            entry_price = close[i];
            stop_loss = entry_price - stop_distance;
            take_profit = entry_price + tp_atr * atr[i];
            position = position_size;
            num_trades++;
        }
        
        if (position > 0) {
            if (low[i+1] <= stop_loss) {
                float profit = (stop_loss - entry_price) * position;
                equity += profit;
                wins += (profit > 0 ? 1 : 0);
                position = 0;
            } else if (high[i+1] >= take_profit) {
                float profit = (take_profit - entry_price) * position;
                equity += profit;
                wins += 1;
                position = 0;
            }
            peak = fmax(peak, equity);
            max_drawdown = fmax(max_drawdown, (peak - equity) / peak);
            if (max_drawdown > max_drawdown_limit) break;
        }
    }

    int base_idx = idx * 3;
    results[base_idx] = equity - 10000.0f;  // Netto-Gewinn
    results[base_idx + 1] = max_drawdown;   // Max Drawdown
    results[base_idx + 2] = num_trades > 0 ? (float)wins / num_trades : 0.0f;  // Gewinnrate
}
"""

# Indikatoren berechnen und zwischenspeichern
def calculate_and_cache_indicators(df, gui_callback):
    if os.path.exists('indicators.pkl'):
        gui_callback("Indikatoren aus Cache geladen.\n")
        return pd.read_pickle('indicators.pkl')
    
    n = len(df)
    mf = cl.mem_flags
    buffers = [
        cl.Buffer(context, mf.READ_ONLY, df['close'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, df['high'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, df['low'].nbytes),
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # rsi
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # ema_short
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # ema_long
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # atr
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # momentum
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # avg_gain
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # avg_loss
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes),  # highest_high
        cl.Buffer(context, mf.READ_WRITE, df['close'].nbytes)   # lowest_low
    ]
    
    zero_array = np.zeros(n, dtype=np.float32)
    for buf in buffers[3:]:
        cl.enqueue_copy(queue, buf, zero_array)
    
    cl.enqueue_copy(queue, buffers[0], df['close'].values)
    cl.enqueue_copy(queue, buffers[1], df['high'].values)
    cl.enqueue_copy(queue, buffers[2], df['low'].values)
    
    program = cl.Program(context, kernel_code).build()
    program.calculate_indicators(queue, (n,), None,
                                 buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
                                 buffers[6], buffers[7], np.int32(n), np.int32(14),
                                 np.int32(25), np.int32(75), buffers[8], buffers[9],
                                 buffers[10], buffers[11], np.int32(3)).wait()
    
    indicators = pd.DataFrame(index=df.index)
    for name, buf in zip(['rsi', 'ema_short', 'ema_long', 'atr', 'momentum', 'highest_high', 'lowest_low'],
                         buffers[3:]):
        arr = np.zeros(n, dtype=np.float32)
        cl.enqueue_copy(queue, arr, buf).wait()
        indicators[name] = arr
    
    indicators.to_pickle('indicators.pkl')
    gui_callback("Indikatoren berechnet und zwischengespeichert.\n")
    return indicators

# GPU Simulation mit zwischengespeicherten Indikatoren
def run_gpu_simulation(df, indicators, param_chunk, program, gui_callback):
    n = len(df)
    num_combinations = len(param_chunk)
    mf = cl.mem_flags
    
    buffers = [
        cl.Buffer(context, mf.READ_ONLY, df['close'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, df['high'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, df['low'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['rsi'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['ema_short'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['ema_long'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['atr'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['momentum'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['highest_high'].nbytes),
        cl.Buffer(context, mf.READ_ONLY, indicators['lowest_low'].nbytes),
        cl.Buffer(context, mf.WRITE_ONLY, 3 * num_combinations * np.float32().nbytes)  # results
    ]
    
    # Daten in Buffer kopieren
    cl.enqueue_copy(queue, buffers[0], df['close'].values)
    cl.enqueue_copy(queue, buffers[1], df['high'].values)
    cl.enqueue_copy(queue, buffers[2], df['low'].values)
    cl.enqueue_copy(queue, buffers[3], indicators['rsi'].values)
    cl.enqueue_copy(queue, buffers[4], indicators['ema_short'].values)
    cl.enqueue_copy(queue, buffers[5], indicators['ema_long'].values)
    cl.enqueue_copy(queue, buffers[6], indicators['atr'].values)
    cl.enqueue_copy(queue, buffers[7], indicators['momentum'].values)
    cl.enqueue_copy(queue, buffers[8], indicators['highest_high'].values)
    cl.enqueue_copy(queue, buffers[9], indicators['lowest_low'].values)
    
    # Parameter für Simulation
    param_array = np.array([[p['rsi_os'], p['rsi_ob'], float(p['ema_short_period']), float(p['ema_long_period']),
                             p['sl_atr'], p['sl_percent'], p['sl_points'], p['tp_atr'], p['tp_percent'],
                             p['tp_points'], p['risk_percent'], p['entry_score_threshold'], float(p['lookback'])]
                            for p in param_chunk], dtype=np.float32).ravel()
    params_buffer = cl.Buffer(context, mf.READ_ONLY, param_array.nbytes)
    cl.enqueue_copy(queue, params_buffer, param_array)
    
    # Strategie simulieren
    local_size = min(256, device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
    global_size = ((num_combinations + local_size - 1) // local_size) * local_size
    program.simulate_strategy(queue, (global_size,), (local_size,),
                              buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
                              buffers[6], buffers[7], buffers[8], buffers[9], buffers[10],
                              np.int32(n), np.int32(num_combinations), params_buffer,
                              np.float32(0.2)).wait()  # Drawdown-Limit 20%
    
    result_data = np.zeros(3 * num_combinations, dtype=np.float32)
    cl.enqueue_copy(queue, result_data, buffers[10]).wait()
    return result_data

# Optimierung
def optimize_strategy(df, indicators, gui_callback):
    param_ranges = {
        'rsi_os': range(20, 40, 5), 'rsi_ob': range(60, 80, 5),
        'ema_short_period': range(10, 50, 10), 'ema_long_period': range(50, 150, 20),
        'sl_atr': np.arange(1.0, 5.0, 0.5), 'sl_percent': [2.0], 'sl_points': [50],
        'tp_atr': np.arange(2.0, 8.0, 1.0), 'tp_percent': [4.0], 'tp_points': [100],
        'risk_percent': [1.0], 'entry_score_threshold': [3], 'lookback': range(2, 10, 2)
    }
    
    total_combinations = np.prod([len(v) for v in param_ranges.values()])
    gui_callback(f"Total Kombinationen: {total_combinations}\n")
    
    program = cl.Program(context, kernel_code).build()
    top_10 = []
    processed = 0
    param_iter = product(*param_ranges.values())
    param_keys = list(param_ranges.keys())
    
    while processed < total_combinations:
        chunk_size = adjust_chunk_size(1000)
        param_chunk = [dict(zip(param_keys, next(param_iter))) for _ in range(min(chunk_size, total_combinations - processed))]
        processed += len(param_chunk)
        
        result_data = run_gpu_simulation(df, indicators, param_chunk, program, gui_callback)
        
        for i in range(len(param_chunk)):
            base_idx = i * 3
            net_profit = result_data[base_idx]
            max_drawdown = result_data[base_idx + 1]
            win_rate = result_data[base_idx + 2]
            if not np.isnan(net_profit) and max_drawdown < 0.2:
                top_10.append({'net_profit': net_profit, 'win_rate': win_rate, 'max_drawdown': max_drawdown, 'params': param_chunk[i]})
                top_10.sort(key=lambda x: x['net_profit'] * x['win_rate'], reverse=True)
                top_10 = top_10[:10]
        
        gui_callback(f"Fortschritt: {processed}/{total_combinations} ({(processed/total_combinations)*100:.2f}%)\n")
    
    return top_10

# GUI
class TradingOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Strategy Optimizer")
        self.root.geometry("800x600")
        
        ttk.Label(root, text="Startdatum:").pack(pady=5)
        self.start_date = DateEntry(root, date_pattern='yyyy-mm-dd')
        self.start_date.pack()
        self.start_date.set_date(datetime(2023, 1, 1))
        
        ttk.Label(root, text="Enddatum:").pack(pady=5)
        self.end_date = DateEntry(root, date_pattern='yyyy-mm-dd')
        self.end_date.pack()
        self.end_date.set_date(datetime(2023, 12, 31))
        
        self.load_button = ttk.Button(root, text="Daten laden", command=self.load_data)
        self.load_button.pack(pady=10)
        
        self.optimize_button = ttk.Button(root, text="Optimieren", command=self.optimize, state='disabled')
        self.optimize_button.pack(pady=10)
        
        self.export_button = ttk.Button(root, text="Exportieren", command=self.export_results, state='disabled')
        self.export_button.pack(pady=10)
        
        self.result_text = tk.Text(root, height=20, width=80)
        self.result_text.pack(pady=10)
        
        self.df = None
        self.indicators = None
        self.top_10 = None
    
    def log_to_gui(self, message):
        self.result_text.insert(tk.END, message)
        self.result_text.see(tk.END)
        self.root.update_idletasks()
    
    def load_data(self):
        self.load_button.config(state='disabled')
        start_date = self.start_date.get_date()  # Gibt ein datetime.date-Objekt zurück
        end_date = self.end_date.get_date()      # Gibt ein datetime.date-Objekt zurück
        
        if os.path.exists('ohlc_data.pkl'):
            self.df = pd.read_pickle('ohlc_data.pkl')
            self.log_to_gui("Daten aus Cache geladen.\n")
        else:
            self.df = fetch_data(start_date, end_date, self.log_to_gui)
        
        self.indicators = calculate_and_cache_indicators(self.df, self.log_to_gui)
        self.optimize_button.config(state='normal')
        self.load_button.config(state='normal')
    
    def optimize(self):
        if self.df is None or self.indicators is None:
            messagebox.showwarning("Warnung", "Bitte zuerst Daten laden!")
            return
        
        self.optimize_button.config(state='disabled')
        self.top_10 = optimize_strategy(self.df, self.indicators, self.log_to_gui)
        
        self.log_to_gui("\nTop 10 Kombinationen:\n")
        for i, result in enumerate(self.top_10, 1):
            self.log_to_gui(f"{i}. Net Profit: {result['net_profit']:.2f}, Win Rate: {result['win_rate']:.2%}, "
                            f"Max Drawdown: {result['max_drawdown']:.2%}, Params: {result['params']}\n")
        
        self.export_button.config(state='normal')
        self.optimize_button.config(state='normal')
    
    def export_results(self):
        if not self.top_10:
            messagebox.showwarning("Warnung", "Keine Ergebnisse zum Exportieren!")
            return
        
        with open('optimization_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Net Profit', 'Win Rate', 'Max Drawdown', 'Parameters'])
            for i, result in enumerate(self.top_10, 1):
                writer.writerow([i, result['net_profit'], result['win_rate'], result['max_drawdown'], str(result['params'])])
        self.log_to_gui("Ergebnisse in 'optimization_results.csv' exportiert.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingOptimizerApp(root)
    root.mainloop()
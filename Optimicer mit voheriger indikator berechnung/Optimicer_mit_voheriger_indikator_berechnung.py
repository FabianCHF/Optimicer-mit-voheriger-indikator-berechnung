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
import pickle
import threading
import queue

# OpenCL Setup (unverändert)
platforms = cl.get_platforms()
if not platforms:
    raise RuntimeError("Keine OpenCL-Plattformen gefunden. Bitte installieren Sie OpenCL-Treiber.")
platform = platforms[0]
gpu_devices = platform.get_devices(cl.device_type.GPU)
if not gpu_devices:
    raise RuntimeError("Keine GPU-Geräte gefunden.")
device = gpu_devices[0]
context = cl.Context([device])
cl_queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Speicherüberwachung (unverändert)
def get_available_memory():
    mem = psutil.virtual_memory()
    return mem.available / 1024**3

def adjust_chunk_size(base_size, min_size=1000):
    available_gb = get_available_memory()
    gpu_mem = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) / 1024**3
    max_work_items = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    safe_size = min(base_size, int((min(available_gb * 0.9, gpu_mem * 0.85)) * 1024**3 / (5 * np.float32().nbytes)))
    return max(min_size, min(safe_size, max_work_items * 2))

# Daten von Binance abrufen (unverändert)
def fetch_data(start_date, end_date, gui_callback):
    exchange = ccxt.binance()
    timeframe = '15m'
    symbol = 'ETH/USDT'
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.min.time())
    since = int(start_datetime.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close']].astype(np.float32)
    assert not np.any(np.isnan(df)), "Daten enthalten NaN-Werte!"
    assert not np.any(np.isinf(df)), "Daten enthalten Inf-Werte!"
    data = {'df': df, 'start_date': start_date, 'end_date': end_date}
    with open('ohlc_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    gui_callback(f"Daten geladen: {len(df)} Kerzen\n")
    return df

# OpenCL Kernel (unverändert)
kernel_code = """
__kernel void calculate_indicators(
    __global const float *close, __global const float *high, __global const float *low,
    __global float *rsi, __global float *ema_short, __global float *ema_long,
    __global float *atr, __global float *momentum,
    __global float *avg_gain, __global float *avg_loss,
    __global float *highest_high, __global float *lowest_low,
    int n, int rsi_length, int ema_short_period, int ema_long_period, int lookback,
    int offset, int chunk_size,
    float prev_ema_short, float prev_ema_long, float prev_atr, float prev_avg_gain, float prev_avg_loss) {
    int gid = get_global_id(0);
    int i = gid + offset;
    if (i >= n || gid >= chunk_size) return;

    momentum[i] = i > 0 ? close[i] - close[i-1] : 0.0f;

    float ema_short_alpha = 2.0f / (ema_short_period + 1);
    float ema_long_alpha = 2.0f / (ema_long_period + 1);
    if (i == 0) {
        ema_short[i] = close[i];
        ema_long[i] = close[i];
    } else if (gid == 0) {
        ema_short[i] = ema_short_alpha * close[i] + (1.0f - ema_short_alpha) * prev_ema_short;
        ema_long[i] = ema_long_alpha * close[i] + (1.0f - ema_long_alpha) * prev_ema_long;
    } else {
        ema_short[i] = ema_short_alpha * close[i] + (1.0f - ema_short_alpha) * ema_short[i-1];
        ema_long[i] = ema_long_alpha * close[i] + (1.0f - ema_long_alpha) * ema_long[i-1];
    }

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
        } else if (gid == 0 && i > rsi_length) {
            float delta = close[i] - close[i-1];
            avg_gain[i] = (prev_avg_gain * (rsi_length - 1) + max(delta, 0.0f)) / rsi_length;
            avg_loss[i] = (prev_avg_loss * (rsi_length - 1) + max(-delta, 0.0f)) / rsi_length;
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

    if (i >= 14) {
        float tr = fmax(fmax(high[i] - low[i], fabs(high[i] - close[i-1])), fabs(low[i] - close[i-1]));
        if (i == 14) {
            float sum_tr = 0.0f;
            for (int j = 0; j < 14; j++) {
                sum_tr += fmax(fmax(high[i-j] - low[i-j], fabs(high[i-j] - close[i-j-1])), fabs(low[i-j] - close[i-j-1]));
            }
            atr[i] = sum_tr / 14.0f;
        } else if (gid == 0 && i > 14) {
            atr[i] = (prev_atr * 13.0f + tr) / 14.0f;
        } else {
            atr[i] = (atr[i-1] * 13.0f + tr) / 14.0f;
        }
    } else {
        atr[i] = 0.0f;
    }

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
    results[base_idx] = equity - 10000.0f;
    results[base_idx + 1] = max_drawdown;
    results[base_idx + 2] = num_trades > 0 ? (float)wins / num_trades : 0.0f;
}
"""

# Indikatoren berechnen und zwischenspeichern (unverändert)
def calculate_and_cache_indicators(df, gui_callback, start_date, end_date):
    cache_file = 'indicators.pkl'
    cache_valid = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if (isinstance(cached_data, dict) and 
                'indicators' in cached_data and 
                'start_date' in cached_data and 
                'end_date' in cached_data and
                cached_data['start_date'] == start_date and 
                cached_data['end_date'] == end_date and 
                len(cached_data['indicators']) == len(df)):
                gui_callback("Indikatoren aus Cache geladen.\n")
                return cached_data['indicators']
            else:
                gui_callback("Indikatoren-Cache ungültig (falsches Format oder Zeitbereich), berechne neu...\n")
        except Exception as e:
            gui_callback(f"Fehler beim Laden des Indikatoren-Caches: {e}. Berechne neu...\n")
    
    n = len(df)
    gui_callback(f"Anzahl der Datenpunkte: {n}\n")
    assert len(df['close']) == len(df['high']) == len(df['low']), "Datenlängen stimmen nicht überein!"
    buffer_size = n * np.float32().nbytes
    gui_callback(f"Buffer-Größe: {buffer_size} Bytes\n")
    gui_callback(f"Verfügbarer GPU-Speicher: {device.get_info(cl.device_info.GLOBAL_MEM_SIZE) / 1024**3:.2f} GB\n")
    mf = cl.mem_flags
    
    gui_callback("Buffer werden erstellt...\n")
    buffers = [
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df['close'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df['high'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df['low'].values),
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # rsi
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # ema_short
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # ema_long
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # atr
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # momentum
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # avg_gain
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # avg_loss
        cl.Buffer(context, mf.READ_WRITE, buffer_size),  # highest_high
        cl.Buffer(context, mf.READ_WRITE, buffer_size)   # lowest_low
    ]
    gui_callback("Buffer erfolgreich erstellt.\n")
    
    gui_callback("Initialisiere Ausgabebuffer mit Nullen...\n")
    zero_array = np.zeros(n, dtype=np.float32)
    for buf in buffers[3:]:
        cl.enqueue_copy(cl_queue, buf, zero_array).wait()
    
    gui_callback("Kompiliere OpenCL-Programm...\n")
    program = cl.Program(context, kernel_code).build()
    gui_callback("OpenCL-Programm erfolgreich kompiliert.\n")
    
    gui_callback("Führe Indikatoren-Kernel in Chunks aus...\n")
    max_work_items = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)[0]
    chunk_size = min(max_work_items * 2, n)
    
    prev_ema_short = df['close'].iloc[0] if n > 0 else 0.0
    prev_ema_long = df['close'].iloc[0] if n > 0 else 0.0
    prev_atr = 0.0
    prev_avg_gain = 0.0
    prev_avg_loss = 0.0
    
    for offset in range(0, n, chunk_size):
        chunk_n = min(chunk_size, n - offset)
        program.calculate_indicators(cl_queue, (chunk_n,), None,
                                     buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
                                     buffers[6], buffers[7], buffers[8], buffers[9], buffers[10], buffers[11],
                                     np.int32(n), np.int32(14), np.int32(25), np.int32(75), np.int32(3),
                                     np.int32(offset), np.int32(chunk_n),
                                     np.float32(prev_ema_short), np.float32(prev_ema_long), np.float32(prev_atr),
                                     np.float32(prev_avg_gain), np.float32(prev_avg_loss)).wait()
        
        if offset + chunk_n < n:
            temp_arr = np.zeros(chunk_n, dtype=np.float32)
            cl.enqueue_copy(cl_queue, temp_arr, buffers[4], offset * 4).wait()
            prev_ema_short = temp_arr[-1]
            cl.enqueue_copy(cl_queue, temp_arr, buffers[5], offset * 4).wait()
            prev_ema_long = temp_arr[-1]
            cl.enqueue_copy(cl_queue, temp_arr, buffers[6], offset * 4).wait()
            prev_atr = temp_arr[-1]
            cl.enqueue_copy(cl_queue, temp_arr, buffers[8], offset * 4).wait()
            prev_avg_gain = temp_arr[-1]
            cl.enqueue_copy(cl_queue, temp_arr, buffers[9], offset * 4).wait()
            prev_avg_loss = temp_arr[-1]
    
    gui_callback("Kernel für alle Chunks erfolgreich ausgeführt.\n")
    
    indicators = pd.DataFrame(index=df.index)
    gui_callback("Kopiere Ergebnisse zurück...\n")
    for name, buf in zip(['rsi', 'ema_short', 'ema_long', 'atr', 'momentum', 'highest_high', 'lowest_low'],
                         buffers[3:8] + buffers[10:12]):
        arr = np.zeros(n, dtype=np.float32)
        cl.enqueue_copy(cl_queue, arr, buf).wait()
        indicators[name] = arr
    
    data = {'indicators': indicators, 'start_date': start_date, 'end_date': end_date}
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    gui_callback("Indikatoren berechnet und zwischengespeichert.\n")
    return indicators

# GPU Simulation (unverändert)
def run_gpu_simulation(df, indicators, param_chunk, program, gui_callback):
    n = len(df)
    num_combinations = len(param_chunk)
    gui_callback(f"Simuliere {num_combinations} Kombinationen mit {n} Datenpunkten.\n")
    mf = cl.mem_flags

    buffers = [
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df['close'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df['high'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df['low'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['rsi'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['ema_short'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['ema_long'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['atr'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['momentum'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['highest_high'].values),
        cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indicators['lowest_low'].values),
        cl.Buffer(context, mf.WRITE_ONLY, 3 * num_combinations * np.float32().nbytes)
    ]

    param_array = np.array([[p['rsi_os'], p['rsi_ob'], float(p['ema_short_period']), float(p['ema_long_period']),
                             p['sl_atr'], p['sl_percent'], p['sl_points'], p['tp_atr'], p['tp_percent'],
                             p['tp_points'], p['risk_percent'], p['entry_score_threshold'], float(p['lookback'])]
                            for p in param_chunk], dtype=np.float32).ravel()
    params_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=param_array)

    local_size = min(512, device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
    global_size = ((num_combinations + local_size - 1) // local_size) * local_size
    event = program.simulate_strategy(cl_queue, (global_size,), (local_size,),
                                      buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
                                      buffers[6], buffers[7], buffers[8], buffers[9], buffers[10],
                                      np.int32(n), np.int32(num_combinations), params_buffer,
                                      np.float32(0.2))
    event.wait()

    result_data = np.zeros(3 * num_combinations, dtype=np.float32)
    cl.enqueue_copy(cl_queue, result_data, buffers[10]).wait()
    gui_callback("Simulation abgeschlossen.\n")
    return result_data

# Optimierung mit Pause-Funktion (unverändert)
def optimize_strategy(df, indicators, gui_callback, result_queue, stop_event, continue_event):
    param_ranges = {
        'rsi_os': range(25, 35, 1),          
        'rsi_ob': range(65, 75, 1),           
        'ema_short_period': range(10, 50, 10),
        'ema_long_period': range(50, 150, 10),
        'sl_atr': np.arange(0.2, 3.0, 0.2),   
        'sl_percent': [0.0],                  # Nicht verwendet, Platzhalter
        'sl_points': [0.0],                   # Nicht verwendet, Platzhalter
        'tp_atr': np.arange(0.2, 6.0, 0.2),   
        'tp_percent': [0.0],                  # Nicht verwendet, Platzhalter
        'tp_points': [0.0],                   # Nicht verwendet, Platzhalter
        'risk_percent': [0.2, 5.0, 0.2],      
        'entry_score_threshold': range(1, 10, 1), 
        'lookback': range(1, 6, 1)           
    }

    total_combinations = np.prod([len(v) for v in param_ranges.values()])
    gui_callback(f"Total Kombinationen: {total_combinations}\n")

    program = cl.Program(context, kernel_code).build()
    top_10 = []
    processed = 0
    param_iter = product(*param_ranges.values())
    param_keys = list(param_ranges.keys())

    while processed < total_combinations:
        if stop_event.is_set():
            gui_callback("Optimierung pausiert...\n")
            result_queue.put(top_10)
            continue_event.wait()
            continue_event.clear()
            gui_callback("Optimierung wird fortgesetzt...\n")

        chunk_size = adjust_chunk_size(2000)
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
    
    result_queue.put(top_10)

# GUI mit TradingView-Optik
class TradingOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Strategy Optimizer")
        self.root.geometry("900x700")
        self.root.configure(bg="#212121")  # TradingView Dunkelgrau

        # Stilkonfiguration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#212121", foreground="#FFFFFF", font=("Roboto", 12))
        style.configure("TButton", background="#424242", foreground="#FFFFFF", font=("Roboto", 11, "bold"), 
                        borderwidth=0, padding=10, relief="flat")
        style.map("TButton", background=[("active", "#616161"), ("disabled", "#333333")])

        # Hauptframe mit abgerundeten Ecken
        main_frame = tk.Frame(self.root, bg="#212121", bd=0, highlightthickness=0)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Titel
        title_label = ttk.Label(main_frame, text="Strategy Optimizer", font=("Roboto", 18, "bold"))
        title_label.pack(pady=(0, 20))

        # Eingabefelder
        input_frame = tk.Frame(main_frame, bg="#212121")
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="Startdatum:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.start_date = DateEntry(input_frame, date_pattern='yyyy-mm-dd', background="#424242", 
                                    foreground="#FFFFFF", borderwidth=0, font=("Roboto", 11))
        self.start_date.grid(row=0, column=1, pady=5)
        self.start_date.set_date(datetime(2023, 1, 1))

        ttk.Label(input_frame, text="Enddatum:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.end_date = DateEntry(input_frame, date_pattern='yyyy-mm-dd', background="#424242", 
                                  foreground="#FFFFFF", borderwidth=0, font=("Roboto", 11))
        self.end_date.grid(row=1, column=1, pady=5)
        self.end_date.set_date(datetime(2023, 12, 31))

        # Buttons
        button_frame = tk.Frame(main_frame, bg="#212121")
        button_frame.pack(pady=20)

        self.load_button = ttk.Button(button_frame, text="Daten laden", command=self.load_data, 
                                      style="TButton", width=15)
        self.load_button.grid(row=0, column=0, padx=10)

        self.optimize_button = ttk.Button(button_frame, text="Optimieren", command=self.start_optimization, 
                                          style="TButton", width=15, state='disabled')
        self.optimize_button.grid(row=0, column=1, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_optimization, 
                                      style="TButton", width=15, state='disabled')
        self.stop_button.grid(row=0, column=2, padx=10)

        self.continue_button = ttk.Button(button_frame, text="Weiterlaufen", command=self.continue_optimization, 
                                          style="TButton", width=15, state='disabled')
        self.continue_button.grid(row=0, column=3, padx=10)

        self.export_button = ttk.Button(button_frame, text="Exportieren", command=self.export_results, 
                                        style="TButton", width=15, state='disabled')
        self.export_button.grid(row=0, column=4, padx=10)

        # Textfeld mit Scrollbar und TradingView-Stil
        text_frame = tk.Frame(main_frame, bg="#212121")
        text_frame.pack(pady=20, fill="both", expand=True)

        self.result_text = tk.Text(text_frame, height=20, width=80, bg="#2E2E2E", fg="#FFFFFF", 
                                   font=("Roboto", 10), bd=0, highlightthickness=0, wrap="word")
        self.result_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=scrollbar.set)

        # Variablen
        self.df = None
        self.indicators = None
        self.top_10 = None
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.continue_event = threading.Event()
        self.opt_thread = None

    def log_to_gui(self, message):
        self.result_text.insert(tk.END, message)
        self.result_text.see(tk.END)
        self.root.update_idletasks()

    def load_data(self):
        self.load_button.config(state='disabled')
        start_date = self.start_date.get_date()
        end_date = self.end_date.get_date()
        
        ohlc_cache_file = 'ohlc_data.pkl'
        cache_valid = False
        if os.path.exists(ohlc_cache_file):
            try:
                with open(ohlc_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if (isinstance(cached_data, dict) and 
                    'df' in cached_data and 
                    'start_date' in cached_data and 
                    'end_date' in cached_data and
                    cached_data['start_date'] == start_date and 
                    cached_data['end_date'] == end_date):
                    self.df = cached_data['df']
                    self.log_to_gui("Daten aus Cache geladen.\n")
                    cache_valid = True
                else:
                    self.log_to_gui("Cache ungültig (falsches Format oder Zeitbereich), lade neue Daten...\n")
            except Exception as e:
                self.log_to_gui(f"Fehler beim Laden des Caches: {e}. Lade neue Daten...\n")
        else:
            self.log_to_gui("Kein Cache gefunden, lade neue Daten...\n")
        
        if not cache_valid:
            self.df = fetch_data(start_date, end_date, self.log_to_gui)
        
        self.indicators = calculate_and_cache_indicators(self.df, self.log_to_gui, start_date, end_date)
        self.optimize_button.config(state='normal')
        self.load_button.config(state='normal')

    def start_optimization(self):
        if self.df is None or self.indicators is None:
            messagebox.showwarning("Warnung", "Bitte zuerst Daten laden!")
            return
        
        self.optimize_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.continue_button.config(state='disabled')
        self.log_to_gui("Optimierung gestartet...\n")
        
        self.stop_event.clear()
        self.continue_event.clear()
        self.opt_thread = threading.Thread(target=optimize_strategy, 
                                           args=(self.df, self.indicators, self.log_to_gui, self.result_queue, 
                                                 self.stop_event, self.continue_event))
        self.opt_thread.start()
        self.root.after(100, self.check_optimization_results)

    def stop_optimization(self):
        if self.opt_thread and self.opt_thread.is_alive():
            self.stop_event.set()
            self.stop_button.config(state='disabled')
            self.continue_button.config(state='normal')
            self.log_to_gui("Stoppen der Optimierung angefordert...\n")

    def continue_optimization(self):
        if self.opt_thread and self.opt_thread.is_alive():
            self.continue_event.set()
            self.stop_button.config(state='normal')
            self.continue_button.config(state='disabled')
            self.log_to_gui("Optimierung wird fortgesetzt...\n")

    def check_optimization_results(self):
        try:
            self.top_10 = self.result_queue.get_nowait()
            self.log_to_gui("\nAktuelle Top 10 Kombinationen:\n")
            for i, result in enumerate(self.top_10, 1):
                self.log_to_gui(f"{i}. Net Profit: {result['net_profit']:.2f}, Win Rate: {result['win_rate']:.2%}, "
                                f"Max Drawdown: {result['max_drawdown']:.2%}, Params: {result['params']}\n")
            self.export_button.config(state='normal')
            if not self.opt_thread.is_alive():
                self.optimize_button.config(state='normal')
                self.stop_button.config(state='disabled')
                self.continue_button.config(state='disabled')
        except queue.Empty:
            self.root.after(100, self.check_optimization_results)

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
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    root = tk.Tk()
    app = TradingOptimizerApp(root)
    root.mainloop()
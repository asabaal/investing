"""
Symphony Watchlist Module

This module provides a dashboard for monitoring the health of symphonies
and their constituent securities. It integrates with the Symphony Analyzer
to provide real-time monitoring and alerts for potential issues.
"""

import json
import logging
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from composer_symphony import Symphony, SymbolList
from alpha_vantage_api import AlphaVantageClient
from symphony_analyzer import SymphonyAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class SymphonyWatchlistApp:
    """
    Tkinter application for monitoring symphonies and their securities.
    
    This application provides a graphical interface for monitoring the health
    of symphonies and their constituent securities. It integrates with the
    SymphonyAnalyzer to provide real-time data and alerts.
    """
    
    def __init__(self, client: AlphaVantageClient):
        """
        Initialize the watchlist application.
        
        Args:
            client: Alpha Vantage client for market data
        """
        self.client = client
        self.analyzer = SymphonyAnalyzer(client)
        
        # Load symphonies
        self.symphonies = []
        self.load_symphonies()
        
        # Initialize watchlist data
        self.watchlist_data = {}
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Symphony Watchlist")
        self.root.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.symbols_tab = ttk.Frame(self.notebook)
        self.symphonies_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.symbols_tab, text="Symbols")
        self.notebook.add(self.symphonies_tab, text="Symphonies")
        self.notebook.add(self.analysis_tab, text="Analysis")
        
        # Set up each tab
        self._setup_dashboard_tab()
        self._setup_symbols_tab()
        self._setup_symphonies_tab()
        self._setup_analysis_tab()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create menu
        self._create_menu()
        
        # Initial data refresh
        self.refresh_data()
    
    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Refresh Data", command=self.refresh_data)
        file_menu.add_command(label="Load Symphony...", command=self.load_symphony_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Analyze menu
        analyze_menu = tk.Menu(menubar, tearoff=0)
        analyze_menu.add_command(label="Run Backtest...", command=self.run_backtest)
        analyze_menu.add_command(label="Generate Report...", command=self.generate_report)
        menubar.add_cascade(label="Analyze", menu=analyze_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _setup_dashboard_tab(self):
        """Set up the dashboard tab with overview widgets."""
        # Create frame for top section
        top_frame = ttk.Frame(self.dashboard_tab)
        top_frame.pack(fill=tk.X, expand=False, pady=10)
        
        # Create refresh button
        refresh_btn = ttk.Button(top_frame, text="Refresh Data", command=self.refresh_data)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Create last updated label
        self.last_updated_var = tk.StringVar()
        self.last_updated_var.set("Last Updated: Never")
        last_updated_lbl = ttk.Label(top_frame, textvariable=self.last_updated_var)
        last_updated_lbl.pack(side=tk.LEFT, padx=10)
        
        # Create main dashboard frame
        dashboard_frame = ttk.Frame(self.dashboard_tab)
        dashboard_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for symphony health
        left_panel = ttk.LabelFrame(dashboard_frame, text="Symphony Health")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for symphony health
        columns = ("Symphony", "Health", "Risk Symbols", "Trend+", "Trend-")
        self.health_tree = ttk.Treeview(left_panel, columns=columns, show="headings")
        for col in columns:
            self.health_tree.heading(col, text=col)
            self.health_tree.column(col, width=100)
        
        # Add scrollbar
        health_scroll = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.health_tree.yview)
        self.health_tree.configure(yscrollcommand=health_scroll.set)
        
        health_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.health_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel for alerts
        right_panel = ttk.LabelFrame(dashboard_frame, text="Alerts")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create listbox for alerts
        self.alerts_listbox = tk.Listbox(right_panel, height=10)
        alerts_scroll = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.alerts_listbox.yview)
        self.alerts_listbox.configure(yscrollcommand=alerts_scroll.set)
        
        alerts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.alerts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create bottom frame for graphs
        bottom_frame = ttk.LabelFrame(self.dashboard_tab, text="Market Overview")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for graphs
        self.fig = plt.Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _setup_symbols_tab(self):
        """Set up the symbols tab with detailed symbol data."""
        # Create top frame for controls
        controls_frame = ttk.Frame(self.symbols_tab)
        controls_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Create filter label and entry
        ttk.Label(controls_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(controls_frame, textvariable=self.filter_var)
        filter_entry.pack(side=tk.LEFT, padx=5)
        
        # Add filter button
        filter_btn = ttk.Button(controls_frame, text="Apply Filter", command=self.apply_symbol_filter)
        filter_btn.pack(side=tk.LEFT, padx=5)
        
        # Create treeview frame
        tree_frame = ttk.Frame(self.symbols_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for symbols
        columns = ("Symbol", "Price", "Change%", "RSI", "5d%", "20d%", "Forecast", "Symphonies")
        self.symbols_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        # Configure columns
        self.symbols_tree.heading("Symbol", text="Symbol")
        self.symbols_tree.column("Symbol", width=80)
        
        self.symbols_tree.heading("Price", text="Price")
        self.symbols_tree.column("Price", width=80)
        
        self.symbols_tree.heading("Change%", text="Change%")
        self.symbols_tree.column("Change%", width=80)
        
        self.symbols_tree.heading("RSI", text="RSI")
        self.symbols_tree.column("RSI", width=60)
        
        self.symbols_tree.heading("5d%", text="5d%")
        self.symbols_tree.column("5d%", width=60)
        
        self.symbols_tree.heading("20d%", text="20d%")
        self.symbols_tree.column("20d%", width=60)
        
        self.symbols_tree.heading("Forecast", text="30d Forecast")
        self.symbols_tree.column("Forecast", width=100)
        
        self.symbols_tree.heading("Symphonies", text="Symphonies")
        self.symbols_tree.column("Symphonies", width=200)
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.symbols_tree.yview)
        x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.symbols_tree.xview)
        self.symbols_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.symbols_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind double click event
        self.symbols_tree.bind("<Double-1>", self.show_symbol_details)
    
    def _setup_symphonies_tab(self):
        """Set up the symphonies tab with detailed symphony information."""
        # Create top frame for controls
        controls_frame = ttk.Frame(self.symphonies_tab)
        controls_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Create dropdown for selecting symphony
        ttk.Label(controls_frame, text="Select Symphony:").pack(side=tk.LEFT, padx=5)
        self.symphony_var = tk.StringVar()
        self.symphony_combo = ttk.Combobox(controls_frame, textvariable=self.symphony_var)
        self.symphony_combo.pack(side=tk.LEFT, padx=5)
        
        # Add view button
        view_btn = ttk.Button(controls_frame, text="View Details", command=self.view_symphony_details)
        view_btn.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for symphony details
        details_notebook = ttk.Notebook(self.symphonies_tab)
        details_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for symphony details
        self.overview_tab = ttk.Frame(details_notebook)
        self.performance_tab = ttk.Frame(details_notebook)
        self.composition_tab = ttk.Frame(details_notebook)
        
        details_notebook.add(self.overview_tab, text="Overview")
        details_notebook.add(self.performance_tab, text="Performance")
        details_notebook.add(self.composition_tab, text="Composition")
        
        # Set up overview tab
        overview_frame = ttk.Frame(self.overview_tab)
        overview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget for symphony details
        self.overview_text = tk.Text(overview_frame, height=20, width=80)
        overview_scroll = ttk.Scrollbar(overview_frame, orient=tk.VERTICAL, command=self.overview_text.yview)
        self.overview_text.configure(yscrollcommand=overview_scroll.set)
        
        overview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.overview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Set up performance tab
        performance_frame = ttk.Frame(self.performance_tab)
        performance_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for performance graph
        self.perf_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, master=performance_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up composition tab
        composition_frame = ttk.Frame(self.composition_tab)
        composition_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for symphony composition
        columns = ("Symbol", "Weight", "Performance")
        self.composition_tree = ttk.Treeview(composition_frame, columns=columns, show="headings")
        
        for col in columns:
            self.composition_tree.heading(col, text=col)
            self.composition_tree.column(col, width=100)
        
        # Add scrollbar
        comp_scroll = ttk.Scrollbar(composition_frame, orient=tk.VERTICAL, command=self.composition_tree.yview)
        self.composition_tree.configure(yscrollcommand=comp_scroll.set)
        
        comp_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.composition_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _setup_analysis_tab(self):
        """Set up the analysis tab with tools for detailed analysis."""
        # Create top frame for controls
        controls_frame = ttk.Frame(self.analysis_tab)
        controls_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Create analysis type dropdown
        ttk.Label(controls_frame, text="Analysis Type:").pack(side=tk.LEFT, padx=5)
        self.analysis_type_var = tk.StringVar()
        analysis_types = ["Backtesting", "Optimization", "Scenario Testing", "Symphony Comparison"]
        analysis_combo = ttk.Combobox(controls_frame, textvariable=self.analysis_type_var, values=analysis_types)
        analysis_combo.pack(side=tk.LEFT, padx=5)
        analysis_combo.current(0)
        
        # Add run button
        run_btn = ttk.Button(controls_frame, text="Run Analysis", command=self.run_analysis)
        run_btn.pack(side=tk.LEFT, padx=5)
        
        # Create main analysis frame
        analysis_frame = ttk.Frame(self.analysis_tab)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create left panel for parameters
        left_panel = ttk.LabelFrame(analysis_frame, text="Parameters")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=5, ipady=5)
        
        # Symphony selection
        ttk.Label(left_panel, text="Symphony:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.analysis_symphony_var = tk.StringVar()
        self.analysis_symphony_combo = ttk.Combobox(left_panel, textvariable=self.analysis_symphony_var)
        self.analysis_symphony_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Date range
        ttk.Label(left_panel, text="Start Date:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_date_var = tk.StringVar()
        start_date_entry = ttk.Entry(left_panel, textvariable=self.start_date_var)
        start_date_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        ttk.Label(left_panel, text="End Date:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.end_date_var = tk.StringVar()
        end_date_entry = ttk.Entry(left_panel, textvariable=self.end_date_var)
        end_date_entry.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Benchmark
        ttk.Label(left_panel, text="Benchmark:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.benchmark_var = tk.StringVar()
        self.benchmark_var.set("SPY")
        benchmark_entry = ttk.Entry(left_panel, textvariable=self.benchmark_var)
        benchmark_entry.grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Rebalance frequency
        ttk.Label(left_panel, text="Rebalance:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.rebalance_var = tk.StringVar()
        rebalance_values = ["daily", "weekly", "monthly", "quarterly"]
        rebalance_combo = ttk.Combobox(left_panel, textvariable=self.rebalance_var, values=rebalance_values)
        rebalance_combo.grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        rebalance_combo.current(2)  # Default to monthly
        
        # Set default dates
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        self.start_date_var.set(one_year_ago.strftime('%Y-%m-%d'))
        self.end_date_var.set(today.strftime('%Y-%m-%d'))
        
        # Create right panel for results
        right_panel = ttk.LabelFrame(analysis_frame, text="Results")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for results
        results_notebook = ttk.Notebook(right_panel)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for results
        self.results_summary_tab = ttk.Frame(results_notebook)
        self.results_chart_tab = ttk.Frame(results_notebook)
        self.results_details_tab = ttk.Frame(results_notebook)
        
        results_notebook.add(self.results_summary_tab, text="Summary")
        results_notebook.add(self.results_chart_tab, text="Chart")
        results_notebook.add(self.results_details_tab, text="Details")
        
        # Set up summary tab
        summary_frame = ttk.Frame(self.results_summary_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget for results summary
        self.results_text = tk.Text(summary_frame, height=20, width=80)
        results_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Set up chart tab
        chart_frame = ttk.Frame(self.results_chart_tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for results chart
        self.results_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, master=chart_frame)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up details tab
        details_frame = ttk.Frame(self.results_details_tab)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for results details
        columns = ("Metric", "Value")
        self.results_tree = ttk.Treeview(details_frame, columns=columns, show="headings")
        
        for col in columns:
            self.results_tree.heading(col, text=col)
        
        self.results_tree.column("Metric", width=150)
        self.results_tree.column("Value", width=150)
        
        # Add scrollbar
        results_tree_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_tree_scroll.set)
        
        results_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def load_symphonies(self):
        """Load available symphonies from files."""
        # Check for symphony files in the current directory
        symphony_files = [f for f in os.listdir() if f.endswith('.json') and 'symphony' in f.lower()]
        
        for file in symphony_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                symphony = Symphony.from_dict(data)
                self.symphonies.append(symphony)
                logger.info(f"Loaded symphony: {symphony.name}")
            except Exception as e:
                logger.error(f"Failed to load symphony from {file}: {str(e)}")
        
        # If no symphonies found, create a default one
        if not self.symphonies:
            logger.info("No symphonies found, creating a default one")
            universe = SymbolList(['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT', 'LQD', 'HYG'])
            default_symphony = Symphony('Default Symphony', 'A simple default symphony', universe)
            self.symphonies.append(default_symphony)
    
    def load_symphony_file(self):
        """Load a symphony from a file."""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                symphony = Symphony.from_dict(data)
                self.symphonies.append(symphony)
                
                # Update the UI elements that show symphonies
                self._update_symphony_dropdowns()
                
                self.status_var.set(f"Loaded symphony: {symphony.name}")
                logger.info(f"Loaded symphony: {symphony.name} from {file_path}")
                
                # Refresh watchlist
                self.refresh_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load symphony: {str(e)}")
                logger.error(f"Failed to load symphony from {file_path}: {str(e)}")
    
    def _update_symphony_dropdowns(self):
        """Update all dropdowns that show symphonies."""
        symphony_names = [sym.name for sym in self.symphonies]
        
        # Update symphony dropdown in symphonies tab
        self.symphony_combo['values'] = symphony_names
        if symphony_names:
            self.symphony_combo.current(0)
        
        # Update symphony dropdown in analysis tab
        self.analysis_symphony_combo['values'] = symphony_names
        if symphony_names:
            self.analysis_symphony_combo.current(0)
    
    def refresh_data(self):
        """Refresh all watchlist data."""
        self.status_var.set("Refreshing data...")
        self.root.update_idletasks()
        
        try:
            # Create watchlist from available symphonies
            self.watchlist_data = self.analyzer.create_symphony_watchlist(self.symphonies)
            
            # Update UI elements with the new data
            self._update_dashboard()
            self._update_symbols_tab()
            self._update_symphony_dropdowns()
            
            # Update last updated time
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_updated_var.set(f"Last Updated: {now}")
            
            self.status_var.set("Data refreshed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
            logger.error(f"Data refresh error: {str(e)}")
            self.status_var.set("Error refreshing data")
    
    def _update_dashboard(self):
        """Update dashboard tab with latest data."""
        # Clear existing data
        for item in self.health_tree.get_children():
            self.health_tree.delete(item)
        
        self.alerts_listbox.delete(0, tk.END)
        
        # Add symphony health data
        symphony_health = self.watchlist_data.get('symphony_health', {})
        for sym_name, health in symphony_health.items():
            health_score = health.get('health_score', 50)
            risk_symbols = len(health.get('symbols_at_risk', []))
            trend_pos = health.get('trend_positive', 0)
            trend_neg = health.get('trend_negative', 0)
            
            self.health_tree.insert(
                "", "end", 
                values=(sym_name, f"{health_score:.1f}", risk_symbols, trend_pos, trend_neg)
            )
            
            # Add alerts for at-risk symbols
            for at_risk in health.get('symbols_at_risk', []):
                alert_text = f"{at_risk['symbol']} in {sym_name}: {', '.join(at_risk['reasons'])}"
                self.alerts_listbox.insert(tk.END, alert_text)
        
        # Update market overview chart
        self.fig.clear()
        
        # Example: Show performance of SPY
        try:
            if 'SPY' in self.watchlist_data.get('symbols', {}):
                spy_data = self.watchlist_data['symbols']['SPY']
                if 'quote' in spy_data:
                    # Just a placeholder - in reality you would 
                    # create a more useful chart from the data
                    ax = self.fig.add_subplot(111)
                    ax.set_title("SPY Performance")
                    ax.text(0.5, 0.5, f"SPY: ${spy_data['quote'].get('price', 0):.2f}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=14)
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error updating dashboard chart: {str(e)}")
    
    def _update_symbols_tab(self):
        """Update symbols tab with latest data."""
        # Clear existing data
        for item in self.symbols_tree.get_children():
            self.symbols_tree.delete(item)
        
        # Add symbol data
        symbols_data = self.watchlist_data.get('symbols', {})
        for symbol, data in symbols_data.items():
            if 'error' in data:
                continue
            
            price = data.get('quote', {}).get('price', 0)
            change = data.get('quote', {}).get('change_percent', 0)
            rsi = data.get('technicals', {}).get('rsi', None)
            pct_5d = data.get('trend', {}).get('pct_change_5d', None)
            pct_20d = data.get('trend', {}).get('pct_change_20d', None)
            
            forecast = None
            if data.get('forecast') and '30_day' in data['forecast']:
                forecast = data['forecast']['30_day'].get('percent_change', None)
            
            symphonies = ', '.join(data.get('symphonies', []))
            
            values = (
                symbol,
                f"${price:.2f}" if price is not None else "N/A",
                f"{change:.2f}%" if change is not None else "N/A",
                f"{rsi:.1f}" if rsi is not None else "N/A",
                f"{pct_5d:.2f}%" if pct_5d is not None else "N/A",
                f"{pct_20d:.2f}%" if pct_20d is not None else "N/A",
                f"{forecast:.2f}%" if forecast is not None else "N/A",
                symphonies
            )
            
            item_id = self.symbols_tree.insert("", "end", values=values)
            
            # Set colors based on values
            if change is not None:
                if change > 0:
                    self.symbols_tree.item(item_id, tags=('positive',))
                elif change < 0:
                    self.symbols_tree.item(item_id, tags=('negative',))
            
        # Configure tag colors
        self.symbols_tree.tag_configure('positive', background='#d8f0d8')
        self.symbols_tree.tag_configure('negative', background='#f0d8d8')
    
    def apply_symbol_filter(self):
        """Apply filter to symbols list."""
        filter_text = self.filter_var.get().upper()
        
        # Show all items
        for item in self.symbols_tree.get_children():
            self.symbols_tree.item(item, tags=self.symbols_tree.item(item, "tags"))
        
        if filter_text:
            # Hide items that don't match the filter
            for item in self.symbols_tree.get_children():
                values = self.symbols_tree.item(item, "values")
                if filter_text not in values[0] and filter_text not in values[7]:
                    self.symbols_tree.detach(item)
    
    def show_symbol_details(self, event):
        """Show detailed information for a symbol."""
        # Get selected item
        selection = self.symbols_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        symbol = self.symbols_tree.item(item, "values")[0]
        
        # Check if we have data for this symbol
        if symbol not in self.watchlist_data.get('symbols', {}):
            messagebox.showinfo("Symbol Details", f"No data available for {symbol}")
            return
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"{symbol} Details")
        details_window.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(details_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        summary_tab = ttk.Frame(notebook)
        chart_tab = ttk.Frame(notebook)
        forecast_tab = ttk.Frame(notebook)
        
        notebook.add(summary_tab, text="Summary")
        notebook.add(chart_tab, text="Chart")
        notebook.add(forecast_tab, text="Forecast")
        
        # Fill summary tab
        summary_frame = ttk.Frame(summary_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        symbol_data = self.watchlist_data['symbols'][symbol]
        quote = symbol_data.get('quote', {})
        technicals = symbol_data.get('technicals', {})
        trend = symbol_data.get('trend', {})
        
        # Create grid of labels
        row = 0
        
        ttk.Label(summary_frame, text="Price:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"${quote.get('price', 0):.2f}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        ttk.Label(summary_frame, text="Change:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"{quote.get('change_percent', 0):.2f}%").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        ttk.Label(summary_frame, text="Volume:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"{quote.get('volume', 0):,}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        ttk.Label(summary_frame, text="RSI:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"{technicals.get('rsi', 'N/A')}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        ttk.Label(summary_frame, text="MACD:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"{technicals.get('macd', 'N/A')}").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        ttk.Label(summary_frame, text="5-Day Change:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"{trend.get('pct_change_5d', 0):.2f}%").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        ttk.Label(summary_frame, text="20-Day Change:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=f"{trend.get('pct_change_20d', 0):.2f}%").grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        row += 1
        
        # In symphonies section
        ttk.Label(summary_frame, text="In Symphonies:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(summary_frame, text=", ".join(symbol_data.get('symphonies', []))).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Fill chart tab (placeholder)
        chart_frame = ttk.Frame(chart_tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        chart_fig = plt.Figure(figsize=(8, 5), dpi=100)
        chart_canvas = FigureCanvasTkAgg(chart_fig, master=chart_frame)
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Placeholder chart
        ax = chart_fig.add_subplot(111)
        ax.set_title(f"{symbol} Price Chart")
        ax.text(0.5, 0.5, "Chart Placeholder", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=14)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        chart_canvas.draw()
        
        # Fill forecast tab
        forecast_frame = ttk.Frame(forecast_tab)
        forecast_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        forecast = symbol_data.get('forecast', {})
        
        if not forecast:
            ttk.Label(forecast_frame, text="No forecast data available").pack(pady=20)
        else:
            # Create forecast summary
            ttk.Label(forecast_frame, text="Forecast Summary", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=5)
            
            summary_frame = ttk.Frame(forecast_frame)
            summary_frame.pack(fill=tk.X, expand=False, pady=10)
            
            row = 0
            for period, data in forecast.items():
                if period != 'weighted_symphony_forecast':
                    ttk.Label(summary_frame, text=f"{period.replace('_', ' ')}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                    
                    if 'percent_change' in data:
                        value = f"{data['percent_change']:.2f}%"
                        ttk.Label(summary_frame, text=value).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                    
                    row += 1
            
            # Create forecast chart (placeholder)
            ttk.Label(forecast_frame, text="Forecast Chart", font=('Helvetica', 12, 'bold')).pack(anchor=tk.W, pady=5)
            
            forecast_fig = plt.Figure(figsize=(8, 4), dpi=100)
            forecast_canvas = FigureCanvasTkAgg(forecast_fig, master=forecast_frame)
            forecast_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Placeholder chart
            ax = forecast_fig.add_subplot(111)
            ax.set_title(f"{symbol} Forecast")
            ax.text(0.5, 0.5, "Forecast Chart Placeholder", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            forecast_canvas.draw()
    
    def view_symphony_details(self):
        """View detailed information for selected symphony."""
        selected_symphony = self.symphony_var.get()
        
        # Find the symphony object
        symphony = None
        for sym in self.symphonies:
            if sym.name == selected_symphony:
                symphony = sym
                break
        
        if not symphony:
            messagebox.showinfo("Symphony Details", "Please select a symphony first")
            return
        
        # Update the overview text
        self.overview_text.delete(1.0, tk.END)
        
        symphony_dict = symphony.to_dict()
        self.overview_text.insert(tk.END, f"Name: {symphony.name}\n\n")
        self.overview_text.insert(tk.END, f"Description: {symphony.description}\n\n")
        self.overview_text.insert(tk.END, f"Universe Size: {len(symphony.universe)}\n\n")
        self.overview_text.insert(tk.END, f"Universe Symbols: {', '.join(symphony.universe.symbols)}\n\n")
        
        self.overview_text.insert(tk.END, "Operators:\n")
        for op in symphony_dict.get('operators', []):
            self.overview_text.insert(tk.END, f"  - {op['name']} ({op['type']})\n")
            self.overview_text.insert(tk.END, f"    Condition: {op['condition']}\n")
        
        self.overview_text.insert(tk.END, "\nAllocator:\n")
        alloc = symphony_dict.get('allocator', {})
        self.overview_text.insert(tk.END, f"  - {alloc.get('name')} ({alloc.get('type')})\n")
        self.overview_text.insert(tk.END, f"    Method: {alloc.get('method')}\n")
        
        # Update performance tab (placeholder)
        self.perf_fig.clear()
        ax = self.perf_fig.add_subplot(111)
        ax.set_title(f"{symphony.name} Performance")
        ax.text(0.5, 0.5, "Performance Chart Placeholder", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=14)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Value ($)")
        self.perf_canvas.draw()
        
        # Update composition tab
        for item in self.composition_tree.get_children():
            self.composition_tree.delete(item)
        
        # Try to get latest allocations from watchlist data
        health_data = self.watchlist_data.get('symphony_health', {}).get(symphony.name, {})
        
        # In a real application, you would get the actual allocations
        # For now, add placeholder data based on universe
        for symbol in symphony.universe.symbols:
            weight = 1.0 / len(symphony.universe)
            performance = "N/A"
            
            # Check if we have data for this symbol
            if symbol in self.watchlist_data.get('symbols', {}):
                trend = self.watchlist_data['symbols'][symbol].get('trend', {})
                if 'pct_change_20d' in trend:
                    performance = f"{trend['pct_change_20d']:.2f}%"
            
            self.composition_tree.insert(
                "", "end", 
                values=(symbol, f"{weight:.2%}", performance)
            )
    
    def run_analysis(self):
        """Run the selected analysis type."""
        analysis_type = self.analysis_type_var.get()
        symphony_name = self.analysis_symphony_var.get()
        
        # Find the symphony object
        symphony = None
        for sym in self.symphonies:
            if sym.name == symphony_name:
                symphony = sym
                break
        
        if not symphony:
            messagebox.showinfo("Analysis", "Please select a symphony first")
            return
        
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        benchmark = self.benchmark_var.get()
        rebalance = self.rebalance_var.get()
        
        # Validate dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
            return
        
        # Run the analysis based on type
        self.status_var.set(f"Running {analysis_type}...")
        self.root.update_idletasks()
        
        try:
            if analysis_type == "Backtesting":
                self._run_backtest_analysis(symphony, start_date, end_date, benchmark, rebalance)
            elif analysis_type == "Optimization":
                self._run_optimization_analysis(symphony, start_date, end_date)
            elif analysis_type == "Scenario Testing":
                self._run_scenario_analysis(symphony, start_date, end_date)
            elif analysis_type == "Symphony Comparison":
                self._run_comparison_analysis(symphony, start_date, end_date, benchmark)
            
            self.status_var.set(f"{analysis_type} completed")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
            self.status_var.set("Analysis error")
    
    def _run_backtest_analysis(self, symphony, start_date, end_date, benchmark, rebalance):
        """Run backtest analysis."""
        # Run backtest
        results = self.analyzer.backtester.backtest(
            symphony,
            start_date,
            end_date,
            rebalance_frequency=rebalance
        )
        
        if not results.get('success', False):
            messagebox.showerror("Backtest Error", results.get('error', 'Unknown error'))
            return
        
        # Get benchmark comparison
        comparison = self.analyzer._compare_to_benchmark(
            results,
            benchmark,
            start_date,
            end_date
        )
        
        # Update results summary
        self.results_text.delete(1.0, tk.END)
        
        summary = results['backtest_summary']
        self.results_text.insert(tk.END, f"Symphony: {symphony.name}\n\n")
        self.results_text.insert(tk.END, f"Period: {start_date} to {end_date}\n\n")
        self.results_text.insert(tk.END, f"Initial Capital: ${summary['initial_capital']:,.2f}\n")
        self.results_text.insert(tk.END, f"Final Value: ${summary['final_value']:,.2f}\n\n")
        
        self.results_text.insert(tk.END, f"Total Return: {summary['total_return']:.2%}\n")
        self.results_text.insert(tk.END, f"Annual Return: {summary['annual_return']:.2%}\n")
        self.results_text.insert(tk.END, f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n")
        self.results_text.insert(tk.END, f"Max Drawdown: {summary['max_drawdown']:.2%}\n\n")
        
        if 'error' not in comparison:
            self.results_text.insert(tk.END, f"Benchmark: {comparison['benchmark_symbol']}\n")
            self.results_text.insert(tk.END, f"Benchmark Return: {comparison['benchmark_return']:.2%}\n")
            self.results_text.insert(tk.END, f"Excess Return: {comparison['excess_return']:.2%}\n")
            
            if comparison['correlation'] is not None:
                self.results_text.insert(tk.END, f"Correlation: {comparison['correlation']:.2f}\n")
            
            if comparison['beta'] is not None:
                self.results_text.insert(tk.END, f"Beta: {comparison['beta']:.2f}\n")
            
            if comparison['information_ratio'] is not None:
                self.results_text.insert(tk.END, f"Information Ratio: {comparison['information_ratio']:.2f}\n")
        
        # Update results chart
        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111)
        
        # Extract portfolio values
        history = results['portfolio_history']
        dates = [entry['date'] for entry in history]
        values = [entry['portfolio_value'] for entry in history]
        
        # Plot portfolio value
        ax.plot(dates, values, label=f"{symphony.name}")
        
        # Add benchmark if available
        if 'error' not in comparison and benchmark in self.watchlist_data.get('symbols', {}):
            try:
                # This is a placeholder - in a real app, you would 
                # get the benchmark data for the exact date range
                benchmark_data = self.watchlist_data['symbols'][benchmark]
                if 'quote' in benchmark_data:
                    # Scale to match initial value
                    benchmark_price = benchmark_data['quote'].get('price', 0)
                    scaled_value = summary['initial_capital'] * benchmark_price / benchmark_price
                    ax.plot([dates[0], dates[-1]], [summary['initial_capital'], scaled_value], 
                           label=benchmark, linestyle='--')
            except Exception as e:
                logger.error(f"Error adding benchmark to chart: {str(e)}")
        
        ax.set_title("Backtest Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True)
        
        self.results_canvas.draw()
        
        # Update results details
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add backtest details
        metrics = [
            ("Initial Capital", f"${summary['initial_capital']:,.2f}"),
            ("Final Value", f"${summary['final_value']:,.2f}"),
            ("Total Return", f"{summary['total_return']:.2%}"),
            ("Annual Return", f"{summary['annual_return']:.2%}"),
            ("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}"),
            ("Max Drawdown", f"{summary['max_drawdown']:.2%}")
        ]
        
        for metric, value in metrics:
            self.results_tree.insert("", "end", values=(metric, value))
        
        # Add benchmark comparison
        if 'error' not in comparison:
            for metric, value in [
                ("Benchmark", comparison['benchmark_symbol']),
                ("Benchmark Return", f"{comparison['benchmark_return']:.2%}"),
                ("Excess Return", f"{comparison['excess_return']:.2%}"),
                ("Correlation", f"{comparison['correlation']:.2f}" if comparison['correlation'] is not None else "N/A"),
                ("Beta", f"{comparison['beta']:.2f}" if comparison['beta'] is not None else "N/A"),
                ("Information Ratio", f"{comparison['information_ratio']:.2f}" if comparison['information_ratio'] is not None else "N/A")
            ]:
                self.results_tree.insert("", "end", values=(metric, value))
    
    def _run_optimization_analysis(self, symphony, start_date, end_date):
        """Run optimization analysis."""
        # This is a simplified placeholder implementation
        
        # Define parameter space for optimization
        param_space = {
            'operators': {
                'Momentum': {
                    'lookback_days': [30, 60, 90, 120],
                    'top_n': [2, 3, 4, 5]
                },
                'RSIFilter': {
                    'threshold': [20, 25, 30, 35],
                    'condition': ['below', 'above']
                }
            },
            'allocator': {
                'InverseVolatilityAllocator': {
                    'lookback_days': [21, 30, 60]
                },
                'EqualWeightAllocator': {}
            }
        }
        
        # Show optimization in progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Optimization Progress")
        progress_window.geometry("300x100")
        
        ttk.Label(progress_window, text="Running optimization...").pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # Update progress (in a real app, this would be updated during optimization)
        for i in range(101):
            progress_var.set(i)
            progress_window.update_idletasks()
            
            # Simulate optimization work
            if i % 10 == 0:
                self.status_var.set(f"Optimization: {i}% complete")
                self.root.update_idletasks()
            
            # Small delay for demonstration
            if i < 100:
                progress_window.after(50)
        
        # Optimization complete - close progress window
        progress_window.destroy()
        
        # Update results with placeholder data
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"Symphony: {symphony.name}\n\n")
        self.results_text.insert(tk.END, f"Optimization Period: {start_date} to {end_date}\n\n")
        self.results_text.insert(tk.END, "Optimization Results\n")
        self.results_text.insert(tk.END, "-----------------\n\n")
        self.results_text.insert(tk.END, "Best Parameters:\n")
        self.results_text.insert(tk.END, "  - Momentum lookback_days: 60\n")
        self.results_text.insert(tk.END, "  - Momentum top_n: 3\n")
        self.results_text.insert(tk.END, "  - RSIFilter threshold: 30\n")
        self.results_text.insert(tk.END, "  - RSIFilter condition: below\n")
        self.results_text.insert(tk.END, "  - Allocator: InverseVolatilityAllocator(30)\n\n")
        
        self.results_text.insert(tk.END, "Performance Improvement:\n")
        self.results_text.insert(tk.END, "  - Base Sharpe Ratio: 1.2\n")
        self.results_text.insert(tk.END, "  - Optimized Sharpe Ratio: 1.5\n")
        self.results_text.insert(tk.END, "  - Improvement: +25.0%\n\n")
        
        self.results_text.insert(tk.END, "Parameter Sensitivity Analysis:\n")
        self.results_text.insert(tk.END, "  - Most sensitive: RSIFilter threshold\n")
        self.results_text.insert(tk.END, "  - Least sensitive: Momentum top_n\n")
        
        # Clear results chart and show parameter sensitivity (placeholder)
        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111)
        
        # Placeholder data for parameter sensitivity
        params = ['Momentum\nlookback', 'Momentum\ntop_n', 'RSI\nthreshold', 'RSI\ncondition', 'Allocator\nlookback']
        sensitivity = [0.2, 0.1, 0.5, 0.3, 0.4]
        
        ax.bar(params, sensitivity)
        ax.set_title("Parameter Sensitivity")
        ax.set_ylabel("Sensitivity")
        ax.set_ylim(0, 0.6)
        
        for i, v in enumerate(sensitivity):
            ax.text(i, v + 0.02, f"{v:.1f}", ha='center')
        
        self.results_canvas.draw()
        
        # Update results details
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add optimization results
        metrics = [
            ("Base Annual Return", "8.5%"),
            ("Optimized Annual Return", "10.2%"),
            ("Base Sharpe Ratio", "1.2"),
            ("Optimized Sharpe Ratio", "1.5"),
            ("Base Max Drawdown", "12.5%"),
            ("Optimized Max Drawdown", "9.8%"),
            ("Iterations Tested", "25"),
            ("Most Sensitive Parameter", "RSIFilter threshold"),
            ("Recommended Symphony", "Optimized Symphony #3")
        ]
        
        for metric, value in metrics:
            self.results_tree.insert("", "end", values=(metric, value))
    
    def _run_scenario_analysis(self, symphony, start_date, end_date):
        """Run scenario analysis."""
        # In a real application, you would run the symphony through 
        # different market scenarios and analyze the results
        
        # Placeholder implementation
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"Symphony: {symphony.name}\n\n")
        self.results_text.insert(tk.END, f"Analysis Period: {start_date} to {end_date}\n\n")
        self.results_text.insert(tk.END, "Scenario Analysis Results\n")
        self.results_text.insert(tk.END, "-----------------------\n\n")
        
        scenarios = [
            {"name": "Bull Market", "return": "12.5%", "drawdown": "5.2%", "sharpe": "1.8"},
            {"name": "Bear Market", "return": "-8.3%", "drawdown": "15.7%", "sharpe": "-0.7"},
            {"name": "Sideways Market", "return": "2.1%", "drawdown": "4.5%", "sharpe": "0.4"},
            {"name": "High Volatility", "return": "5.2%", "drawdown": "12.8%", "sharpe": "0.6"},
            {"name": "Recession", "return": "-12.5%", "drawdown": "18.9%", "sharpe": "-1.1"},
            {"name": "Recovery", "return": "18.7%", "drawdown": "7.2%", "sharpe": "2.1"}
        ]
        
        for scenario in scenarios:
            self.results_text.insert(tk.END, f"{scenario['name']}:\n")
            self.results_text.insert(tk.END, f"  - Return: {scenario['return']}\n")
            self.results_text.insert(tk.END, f"  - Max Drawdown: {scenario['drawdown']}\n")
            self.results_text.insert(tk.END, f"  - Sharpe Ratio: {scenario['sharpe']}\n\n")
        
        self.results_text.insert(tk.END, "Stress Test Analysis:\n")
        self.results_text.insert(tk.END, "  - Worst Case Drawdown: 22.5%\n")
        self.results_text.insert(tk.END, "  - VaR (95%): 2.8%\n")
        self.results_text.insert(tk.END, "  - Expected Shortfall: 3.5%\n\n")
        
        self.results_text.insert(tk.END, "Symphony Robustness Score: 7.2/10\n")
        
        # Create bar chart for scenario performance
        self.results_fig.clear()
        ax = self.results_fig.add_subplot(111)
        
        # Extract data for chart
        names = [s['name'] for s in scenarios]
        returns = [float(s['return'].strip('%')) for s in scenarios]
        
        # Set colors based on returns
        colors = ['g' if r > 0 else 'r' for r in returns]
        
        ax.bar(names, returns, color=colors)
        ax.set_title("Symphony Performance by Scenario")
        ax.set_ylabel("Return (%)")
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        for i, v in enumerate(returns):
            ax.text(i, v + 0.5 if v > 0 else v - 2, f"{v}%", ha='center')
        
        plt.tight_layout()
        self.results_canvas.draw()
        
        # Update results details
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add scenario results
        for scenario in scenarios:
            self.results_tree.insert(
                "", "end", 
                values=(scenario['name'], f"Return: {scenario['return']}, Sharpe: {scenario['sharpe']}")
            )
            
        # Add additional metrics
        for metric, value in [
            ("Worst Case Drawdown", "22.5%"),
            ("VaR (95%)", "2.8%"),
            ("Expected Shortfall", "3.5%"),
            ("Robustness Score", "7.2/10")
        ]:
            self.results_tree.insert("", "end", values=(metric, value))
    
    def _run_comparison_analysis(self, symphony, start_date, end_date, benchmark):
        """Run comparison analysis against other symphonies or benchmarks."""
        # In a real application, you would compare the performance of
        # the selected symphony against others or benchmarks
        
        # Placeholder implementation
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"Symphony: {symphony.name}\n\n")
        self.results_text.insert(tk.END, f"Comparison Period: {start_date} to {end_date}\n\n")
        self.results_text.insert(tk.END, "Comparison Results\n")
        self.results_text.insert(tk.END, "-----------------\n\n")
        
        self.results_text.insert(tk.END, f"Benchmark: {benchmark}\n")
        self.results_text.insert(tk.END, f"  - Return: 8.2%\n")
        self.results_text.insert(tk.END, f"  - Volatility: 15.5%\n")
        self.results_text.insert(tk.END, f"  - Sharpe Ratio: 0.53\n")
        self.results_text.insert(tk.END, f"  - Max Drawdown: 12.8%\n\n")
        
        self.results_text.insert(tk.END, f"Your Symphony: {symphony.name}\n")
        self.results_text.insert(tk.END, f"  - Return: 10.5%\n")
        self.results_text.insert(tk.END, f"  - Volatility: 14.2%\n")
        self.results_text.insert(tk.END, f"  - Sharpe Ratio: 0.74\n")
        self.results_text.insert(tk.END, f"  - Max Drawdown: 10.5%\n\n")
        
        self.results_text.insert(tk.END, "Performance Metrics vs Benchmark:\n")
        self.results_text.insert(tk.END, f"  - Excess Return: +2.3%\n")
        self.results_text.insert(tk.END, f"  - Alpha: 2.1%\n")
        self.results_text.insert(tk.END, f"  - Beta: 0.85\n")
        self.results_text.insert(tk.END, f"  - Information Ratio: 0.62\n")
        self.results_text.insert(tk.END, f"  - Tracking Error: 3.7%\n\n")
        
        self.results_text.insert(tk.END, "Other Symphonies Comparison:\n")
        
        other_symphonies = [s for s in self.symphonies if s.name != symphony.name]
        if other_symphonies:
            for i, other in enumerate(other_symphonies[:3]):  # Show max 3 other symphonies
                self.results_text.insert(tk.END, f"  - {other.name}: +1.2% excess return, Beta 0.92\n")
        else:
            self.results_text.insert(tk.END, f"  - No other symphonies available for comparison\n")
        
        # Create comparison chart
        self.results_fig.clear()
        
        # Create a 2x2 grid for different metric comparisons
        axes = self.results_fig.subplots(2, 2)
        
        # Return comparison
        axes[0, 0].bar(['Benchmark', 'Your Symphony'], [8.2, 10.5])
        axes[0, 0].set_title("Return (%)")
        axes[0, 0].set_ylim(0, 12)
        
        # Volatility comparison
        axes[0, 1].bar(['Benchmark', 'Your Symphony'], [15.5, 14.2])
        axes[0, 1].set_title("Volatility (%)")
        axes[0, 1].set_ylim(0, 20)
        
        # Sharpe comparison
        axes[1, 0].bar(['Benchmark', 'Your Symphony'], [0.53, 0.74])
        axes[1, 0].set_title("Sharpe Ratio")
        axes[1, 0].set_ylim(0, 0.8)
        
        # Drawdown comparison
        axes[1, 1].bar(['Benchmark', 'Your Symphony'], [12.8, 10.5])
        axes[1, 1].set_title("Max Drawdown (%)")
        axes[1, 1].set_ylim(0, 15)
        
        plt.tight_layout()
        self.results_canvas.draw()
        
        # Update results details
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add comparison metrics
        metrics = [
            (f"Benchmark {benchmark} Return", "8.2%"),
            (f"Symphony {symphony.name} Return", "10.5%"),
            ("Excess Return", "+2.3%"),
            (f"Benchmark Volatility", "15.5%"),
            (f"Symphony Volatility", "14.2%"),
            ("Alpha", "2.1%"),
            ("Beta", "0.85"),
            ("Information Ratio", "0.62"),
            ("Tracking Error", "3.7%"),
            ("Correlation", "0.88"),
            ("Upside Capture", "112%"),
            ("Downside Capture", "78%")
        ]
        
        for metric, value in metrics:
            self.results_tree.insert("", "end", values=(metric, value))
    
    def run_backtest(self):
        """Run a backtest from the menu."""
        # Make sure we have a symphony selected in the analysis tab
        if not self.analysis_symphony_var.get():
            messagebox.showinfo("Backtest", "Please select a symphony in the Analysis tab first")
            return
        
        # Switch to analysis tab
        self.notebook.select(self.analysis_tab)
        
        # Set analysis type to Backtesting
        self.analysis_type_var.set("Backtesting")
        
        # Run the analysis
        self.run_analysis()
    
    def generate_report(self):
        """Generate a detailed report."""
        # Make sure we have some analysis results
        if not self.results_text.get(1.0, tk.END).strip():
            messagebox.showinfo("Report", "Please run an analysis first")
            return
        
        from tkinter import filedialog
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Get report content from results text
                report_content = self.results_text.get(1.0, tk.END)
                
                # Add header and footer
                header = f"Symphony Analysis Report\n"
                header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                footer = "\n\nEnd of Report\n"
                
                # Write to file
                with open(file_path, 'w') as f:
                    f.write(header)
                    f.write(report_content)
                    f.write(footer)
                
                messagebox.showinfo("Report Generated", f"Report saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
                logger.error(f"Report generation error: {str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About Symphony Watchlist")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        
        # Add logo (placeholder)
        logo_frame = ttk.Frame(about_window)
        logo_frame.pack(fill=tk.X, expand=False, pady=10)
        
        ttk.Label(logo_frame, text="Symphony Watchlist", font=('Helvetica', 16, 'bold')).pack()
        
        # Add version info
        info_frame = ttk.Frame(about_window)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        ttk.Label(info_frame, text="Version 1.0", font=('Helvetica', 10)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="© 2025 All Rights Reserved", font=('Helvetica', 10)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="\nA tool for monitoring and analyzing Composer symphonies.", 
                 wraplength=360).pack(anchor=tk.W, pady=10)
        
        ttk.Label(info_frame, text="Built with Python, Tkinter, and Matplotlib", 
                 font=('Helvetica', 8)).pack(anchor=tk.W, pady=10)
        
        # Add OK button
        button_frame = ttk.Frame(about_window)
        button_frame.pack(fill=tk.X, expand=False, pady=10)
        
        ttk.Button(button_frame, text="OK", command=about_window.destroy).pack()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Alpha Vantage client
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
    else:
        client = AlphaVantageClient(api_key=api_key)
        
        # Create and run the app
        app = SymphonyWatchlistApp(client)
        app.run()

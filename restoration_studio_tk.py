import os
import shutil
import tempfile
import time
import wave
from pathlib import Path

import pygame
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from restoration_backend import ROOT, RestorationStudioBackend


class RestorationStudioTkApp:
    COLOR_MAP = {
        "slate": "#607D8B",
        "blue": "#1565C0",
        "green": "#2E7D32",
        "orange": "#EF6C00",
        "red": "#C62828",
    }
    APP_BG = "#F4F6F8"
    CARD_BG = "#FFFFFF"
    ORIGINAL_PANEL_BG = "#FBF7F1"
    ENHANCED_PANEL_BG = "#F3F8F7"
    MUTED_TEXT = "#5F6C7B"
    BODY_TEXT = "#1B2430"

    def __init__(self, root):
        self.root = root
        self.root.title("Music Restoration Studio")
        self.root.geometry("1060x780")
        self.root.minsize(960, 720)
        self.root.configure(bg=self.APP_BG)

        self.backend = RestorationStudioBackend(ROOT)
        self.selected_profile = tk.StringVar(value="restore")
        self.download_mode = tk.StringVar(value="direct")
        self.browser_source = tk.StringVar(value=self.backend.browser_source)
        self.export_format = tk.StringVar(value="mp3")
        self.show_details = tk.BooleanVar(value=False)

        self.restoration_cleanup = tk.BooleanVar(value=False)
        self.restoration_preset = tk.StringVar(value=RestorationStudioBackend.RESTORATION_PRESET_LABELS["medium"])
        self.hum_frequency = tk.StringVar(value="60 Hz")
        self.clarity_mastering = tk.BooleanVar(value=False)
        self.normalize_audio = tk.BooleanVar(value=False)
        self.stem_rebalance = tk.BooleanVar(value=False)
        self.bandwidth_restore = tk.BooleanVar(value=False)
        self.backend_choice = tk.StringVar(value="roformer")
        self.bass_boost = tk.IntVar(value=0)
        self.treble_boost = tk.IntVar(value=0)
        self.volume_boost = tk.IntVar(value=0)

        self.playback_kind = None
        self.playback_source = None
        self.playback_cache = {}
        self.is_playing = False
        self.is_paused = False
        self.playback_base_position = 0.0
        self.playback_start_time = None
        self.paused_position = 0.0
        self.song_length = 0.0
        self.player_widgets = {}
        self.seek_dragging_kind = None
        self.seek_segment_paths = {}

        pygame.mixer.init()

        self._configure_styles()
        self._build_ui()
        self._apply_profile_defaults("restore")
        self._refresh_state()
        self._update_playback_position()

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.APP_BG)
        style.configure("Card.TFrame", background=self.CARD_BG, relief="flat")
        style.configure("Header.TLabel", background=self.APP_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 20, "bold"))
        style.configure("Subtle.TLabel", background=self.CARD_BG, foreground=self.MUTED_TEXT, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=self.CARD_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabel", background=self.CARD_BG, foreground=self.MUTED_TEXT, font=("Segoe UI", 9, "bold"))
        style.configure("Body.TLabel", background=self.CARD_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Compact.TRadiobutton", background=self.CARD_BG, font=("Segoe UI", 9))
        style.configure("Compact.TCheckbutton", background=self.CARD_BG, font=("Segoe UI", 9))
        style.configure("TLabelFrame", background=self.CARD_BG)
        style.configure("TLabelFrame.Label", background=self.CARD_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 10, "bold"))

        style.configure("OriginalPanel.TLabelframe", background=self.ORIGINAL_PANEL_BG)
        style.configure("OriginalPanel.TLabelframe.Label", background=self.ORIGINAL_PANEL_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 10, "bold"))
        style.configure("OriginalPanel.TFrame", background=self.ORIGINAL_PANEL_BG)
        style.configure("OriginalPanelBody.TLabel", background=self.ORIGINAL_PANEL_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 10))
        style.configure("OriginalPanelSubtle.TLabel", background=self.ORIGINAL_PANEL_BG, foreground=self.MUTED_TEXT, font=("Segoe UI", 9))

        style.configure("EnhancedPanel.TLabelframe", background=self.ENHANCED_PANEL_BG)
        style.configure("EnhancedPanel.TLabelframe.Label", background=self.ENHANCED_PANEL_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 10, "bold"))
        style.configure("EnhancedPanel.TFrame", background=self.ENHANCED_PANEL_BG)
        style.configure("EnhancedPanelBody.TLabel", background=self.ENHANCED_PANEL_BG, foreground=self.BODY_TEXT, font=("Segoe UI", 10))
        style.configure("EnhancedPanelSubtle.TLabel", background=self.ENHANCED_PANEL_BG, foreground=self.MUTED_TEXT, font=("Segoe UI", 9))

    def _build_ui(self):
        shell = ttk.Frame(self.root, style="App.TFrame", padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=3)
        shell.columnconfigure(1, weight=2)
        shell.rowconfigure(1, weight=1)

        hero = ttk.Frame(shell, style="App.TFrame")
        hero.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(hero, text="Music Restoration Studio", style="Header.TLabel").pack(anchor="w")

        left = ttk.Frame(shell, style="App.TFrame")
        right = ttk.Frame(shell, style="App.TFrame")
        bottom = ttk.Frame(shell, style="App.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(14, 0))
        right.grid(row=1, column=1, sticky="nsew", pady=(14, 0))
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        bottom.columnconfigure(0, weight=1)

        self._build_source_card(left)
        self._build_profile_card(left)
        self._build_actions_card(right)
        self._build_details_card(right)
        self._build_status_card(bottom)

    def _make_card(self, parent, title):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        frame.pack(fill="x", pady=(0, 8))
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        return frame

    def _build_source_card(self, parent):
        card = self._make_card(parent, "Source")

        url_row = ttk.Frame(card, style="Card.TFrame")
        url_row.pack(fill="x")
        url_row.columnconfigure(0, weight=1)
        self.url_entry = ttk.Entry(url_row)
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.download_button = ttk.Button(url_row, text="Download Source", style="Primary.TButton", command=self.start_download)
        self.download_button.grid(row=0, column=1)

        session_row = ttk.Frame(card, style="Card.TFrame")
        session_row.pack(fill="x", pady=(8, 0))
        ttk.Label(session_row, text="Download Session Source", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            session_row, text="Direct", value="direct", variable=self.download_mode, style="Compact.TRadiobutton",
            command=self._sync_download_mode
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Radiobutton(
            session_row, text="Use browser session", value="browser", variable=self.download_mode, style="Compact.TRadiobutton",
            command=self._sync_download_mode
        ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(3, 0))
        ttk.Label(session_row, text="Browser", style="Section.TLabel").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.browser_combo = ttk.Combobox(
            session_row,
            textvariable=self.browser_source,
            values=[label.capitalize() for label in RestorationStudioBackend.BROWSER_OPTIONS],
            state="readonly",
            width=10,
        )
        self.browser_combo.grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(3, 0))
        self.browser_combo.set(self.browser_source.get().capitalize())

        import_row = ttk.Frame(card, style="Card.TFrame")
        import_row.pack(fill="x", pady=(8, 0))
        import_row.columnconfigure(0, weight=1)
        self.import_path_var = tk.StringVar(value="No local file selected")
        ttk.Label(import_row, textvariable=self.import_path_var, style="Subtle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(import_row, text="Import File", command=self.import_file).grid(row=0, column=1, padx=(8, 0))

    def _build_profile_card(self, parent):
        card = self._make_card(parent, "Processing Profile")

        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="Profile", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.profile_combo = ttk.Combobox(
            top,
            state="readonly",
            values=list(RestorationStudioBackend.PROFILE_LABELS.values()),
            width=26,
        )
        self.profile_combo.grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)
        self.profile_combo.set(RestorationStudioBackend.PROFILE_LABELS["restore"])

        toggles = ttk.Frame(card, style="Card.TFrame")
        toggles.pack(fill="x", pady=(8, 0))
        toggles.columnconfigure(0, weight=1)
        toggles.columnconfigure(1, weight=1)

        self.cleanup_check = ttk.Checkbutton(
            toggles,
            text="Noise Restoration",
            variable=self.restoration_cleanup,
            style="Compact.TCheckbutton",
            command=self._sync_profile_rules,
        )
        self.cleanup_check.grid(row=0, column=0, sticky="w")
        self.mastering_check = ttk.Checkbutton(
            toggles, text="Clarity / Mastering", variable=self.clarity_mastering, style="Compact.TCheckbutton"
        )
        self.mastering_check.grid(row=0, column=1, sticky="w")
        self.normalize_check = ttk.Checkbutton(
            toggles, text="Normalize loudness", variable=self.normalize_audio, style="Compact.TCheckbutton"
        )
        self.normalize_check.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.stem_check = ttk.Checkbutton(
            toggles, text="Stem Rebalance", variable=self.stem_rebalance, style="Compact.TCheckbutton",
            command=self._sync_profile_rules
        )
        self.stem_check.grid(row=1, column=1, sticky="w", pady=(4, 0))
        self.bandwidth_check = ttk.Checkbutton(
            toggles, text="Bandwidth Restore (Experimental)", variable=self.bandwidth_restore, style="Compact.TCheckbutton"
        )
        self.bandwidth_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        setup = ttk.Frame(card, style="Card.TFrame")
        setup.pack(fill="x", pady=(8, 0))
        for column in range(4):
            setup.columnconfigure(column, weight=1)
        ttk.Label(setup, text="Restore strength", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(setup, text="Hum frequency", style="Section.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(setup, text="Stem backend", style="Section.TLabel").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Label(setup, text="Export format", style="Section.TLabel").grid(row=0, column=3, sticky="w", padx=(12, 0))
        self.restoration_preset_combo = ttk.Combobox(
            setup,
            textvariable=self.restoration_preset,
            values=[RestorationStudioBackend.RESTORATION_PRESET_LABELS[key] for key in RestorationStudioBackend.RESTORATION_PRESET_ORDER],
            state="readonly",
            width=15,
        )
        self.restoration_preset_combo.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        self.hum_combo = ttk.Combobox(
            setup,
            textvariable=self.hum_frequency,
            values=[f"{freq} Hz" for freq in RestorationStudioBackend.HUM_FREQUENCIES],
            state="readonly",
            width=9,
        )
        self.hum_combo.grid(row=1, column=1, sticky="ew", padx=(12, 0), pady=(3, 0))
        self.backend_combo = ttk.Combobox(
            setup,
            textvariable=self.backend_choice,
            values=["roformer", "demucs_legacy"],
            state="readonly",
            width=13,
        )
        self.backend_combo.grid(row=1, column=2, sticky="ew", padx=(12, 0), pady=(3, 0))
        self.export_combo = ttk.Combobox(
            setup,
            textvariable=self.export_format,
            values=["mp3", "wav", "flac"],
            state="readonly",
            width=8,
        )
        self.export_combo.grid(row=1, column=3, sticky="ew", padx=(12, 0), pady=(3, 0))

        sliders = ttk.Frame(card, style="Card.TFrame")
        sliders.pack(fill="x", pady=(6, 0))
        sliders.columnconfigure(1, weight=1)
        sliders.columnconfigure(3, weight=1)
        sliders.columnconfigure(5, weight=1)

        self._build_slider(sliders, 0, "Bass", self.bass_boost)
        self._build_slider(sliders, 2, "Treble", self.treble_boost)
        self._build_slider(sliders, 4, "Volume", self.volume_boost)

    def _build_slider(self, parent, column, label, variable):
        ttk.Label(parent, text=label, style="Section.TLabel").grid(row=0, column=column, sticky="w", padx=(0, 6))
        scale = tk.Scale(
            parent,
            from_=-10,
            to=10,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=False,
            bg="#FFFFFF",
            highlightthickness=0,
            variable=variable,
            length=108,
        )
        scale.grid(row=0, column=column + 1, sticky="ew", padx=(0, 12))
        value_label = ttk.Label(parent, text="0 dB", style="Subtle.TLabel")
        value_label.grid(row=1, column=column, columnspan=2, sticky="w")
        variable.trace_add("write", lambda *args, lbl=value_label, var=variable: lbl.config(text=f"{var.get()} dB"))

    def _build_status_card(self, parent):
        card = self._make_card(parent, "Session Status")
        self.status_text_var = tk.StringVar(value="Ready to download or import")
        self.status_label = ttk.Label(card, textvariable=self.status_text_var, style="Body.TLabel", wraplength=940, justify="left")
        self.status_label.pack(anchor="w")

        self.progress = ttk.Progressbar(card, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(10, 8))
        self.progress_value_var = tk.StringVar(value="0%")
        ttk.Label(card, textvariable=self.progress_value_var, style="Subtle.TLabel").pack(anchor="w")

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x", pady=(12, 0))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        grid.columnconfigure(3, weight=1)

        self.roformer_var = tk.StringVar(value="Not resolved yet")
        self.original_name_var = tk.StringVar(value="No original source yet")
        self.enhanced_name_var = tk.StringVar(value="No enhanced version yet")
        self.session_dir_var = tk.StringVar(value="No active session")
        self._build_stat(grid, 0, 0, "RoFormer Backend", self.roformer_var)
        self._build_stat(grid, 0, 1, "Original Source", self.original_name_var)
        self._build_stat(grid, 0, 2, "Enhanced Version", self.enhanced_name_var)
        self._build_stat(grid, 0, 3, "Session Folder", self.session_dir_var)

    def _build_stat(self, parent, row, column, title, variable):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        frame.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0), pady=0)
        ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=variable, style="Subtle.TLabel", wraplength=205, justify="left").pack(anchor="w", pady=(6, 0))

    def _build_actions_card(self, parent):
        card = self._make_card(parent, "Actions and Compare")
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.pack(fill="x")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.apply_button = ttk.Button(actions, text="Apply Profile", style="Primary.TButton", command=self.apply_profile)
        self.apply_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.undo_button = ttk.Button(actions, text="Undo Last", command=self.undo_last)
        self.undo_button.grid(row=0, column=1, sticky="ew")
        self.revert_button = ttk.Button(actions, text="Return to Original", command=self.revert_to_original)
        self.revert_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(8, 0))
        self.export_button = ttk.Button(actions, text="Export Current", command=self.export_current)
        self.export_button.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        compare = ttk.Frame(card, style="Card.TFrame")
        compare.pack(fill="x", pady=(14, 0))
        compare.columnconfigure(0, weight=1)
        compare.columnconfigure(1, weight=1)

        original_frame = ttk.LabelFrame(compare, text="Original", padding=10, style="OriginalPanel.TLabelframe")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._build_player_panel(original_frame, "original", "Original source")

        enhanced_frame = ttk.LabelFrame(compare, text="Enhanced", padding=10, style="EnhancedPanel.TLabelframe")
        enhanced_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_player_panel(enhanced_frame, "enhanced", "Latest enhanced version")

    def _build_player_panel(self, parent, kind, fallback_label):
        frame_style = "OriginalPanel.TFrame" if kind == "original" else "EnhancedPanel.TFrame"
        body_style = "OriginalPanelBody.TLabel" if kind == "original" else "EnhancedPanelBody.TLabel"
        subtle_style = "OriginalPanelSubtle.TLabel" if kind == "original" else "EnhancedPanelSubtle.TLabel"
        parent.columnconfigure(0, weight=1)
        name_var = tk.StringVar(value=fallback_label)
        status_var = tk.StringVar(value="Ready to compare")
        time_var = tk.StringVar(value="0:00 / 0:00")
        seek_var = tk.DoubleVar(value=0.0)

        ttk.Label(parent, textvariable=name_var, style=body_style, wraplength=180, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(parent, textvariable=status_var, style=subtle_style).grid(row=1, column=0, sticky="w", pady=(2, 0))

        progress = ttk.Scale(parent, from_=0, to=100, orient="horizontal", variable=seek_var)
        progress.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        progress.bind("<ButtonPress-1>", lambda _event, item=kind: self._begin_seek(item))
        progress.bind("<ButtonRelease-1>", lambda _event, item=kind: self._commit_seek(item))

        footer = ttk.Frame(parent, style=frame_style)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=time_var, style=subtle_style).grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(parent, style=frame_style)
        controls.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        play_button = ttk.Button(controls, text="Play", command=lambda item=kind: self.play_audio(item))
        play_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        pause_button = ttk.Button(controls, text="Pause", command=lambda item=kind: self.pause_audio(item))
        pause_button.grid(row=0, column=1, sticky="ew", padx=4)
        stop_button = ttk.Button(controls, text="Stop", command=lambda item=kind: self.stop_audio(item))
        stop_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self.player_widgets[kind] = {
            "name_var": name_var,
            "status_var": status_var,
            "time_var": time_var,
            "seek_var": seek_var,
            "progress": progress,
            "play_button": play_button,
            "pause_button": pause_button,
            "stop_button": stop_button,
            "fallback_label": fallback_label,
        }

    def _build_details_card(self, parent):
        card = self._make_card(parent, "Diagnostics")
        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="Recent backend details", style="Section.TLabel").pack(side="left")
        self.toggle_details_button = ttk.Button(top, text="Show details", command=self.toggle_details)
        self.toggle_details_button.pack(side="right")

        self.details_frame = ttk.Frame(card, style="Card.TFrame")
        self.details_text = tk.Text(
            self.details_frame,
            height=10,
            wrap="word",
            bg="#0F172A",
            fg="#D8E1EA",
            relief="flat",
            font=("Consolas", 9),
        )
        self.details_text.pack(fill="both", expand=True)
        self.details_text.config(state="disabled")

    def _sync_download_mode(self):
        state = "readonly" if self.download_mode.get() == "browser" else "disabled"
        self.browser_combo.config(state=state)

    def _profile_key_from_label(self, label):
        for key, value in RestorationStudioBackend.PROFILE_LABELS.items():
            if value == label:
                return key
        return "restore"

    def _restoration_preset_key_from_label(self, label):
        for key, value in RestorationStudioBackend.RESTORATION_PRESET_LABELS.items():
            if value == label:
                return key
        return "medium"

    def _restoration_preset_label_from_key(self, key):
        return RestorationStudioBackend.RESTORATION_PRESET_LABELS.get(key, RestorationStudioBackend.RESTORATION_PRESET_LABELS["medium"])

    def _normalize_hum_label(self, value):
        text = str(value).strip().replace("hz", "").replace("Hz", "").strip()
        try:
            freq = int(text)
        except ValueError:
            freq = 60
        if freq not in RestorationStudioBackend.HUM_FREQUENCIES:
            freq = 60
        return f"{freq} Hz"

    def _on_profile_change(self, _event=None):
        profile = self._profile_key_from_label(self.profile_combo.get())
        self._apply_profile_defaults(profile)

    def _apply_profile_defaults(self, profile):
        defaults = RestorationStudioBackend.PROFILE_DEFAULTS[profile]
        self.selected_profile.set(profile)
        self.profile_combo.set(RestorationStudioBackend.PROFILE_LABELS[profile])
        self.restoration_cleanup.set(defaults["restoration_cleanup"])
        self.restoration_preset.set(self._restoration_preset_label_from_key(defaults.get("restoration_preset", "medium")))
        self.hum_frequency.set(self._normalize_hum_label(defaults.get("hum_frequency", 60)))
        self.clarity_mastering.set(defaults["clarity_mastering"])
        self.normalize_audio.set(defaults["normalize_audio"])
        self.stem_rebalance.set(defaults["stem_rebalance"])
        self.bandwidth_restore.set(defaults["bandwidth_restore"])
        self.backend_choice.set(defaults["backend"])
        self.bass_boost.set(defaults["bass_boost"])
        self.treble_boost.set(defaults["treble_boost"])
        self.volume_boost.set(defaults["volume_boost"])
        self._sync_profile_rules()

    def _sync_profile_rules(self):
        profile = self.selected_profile.get()
        advanced = profile == "advanced"
        stem_mode = profile == "stem"

        if stem_mode:
            self.restoration_cleanup.set(False)
            self.clarity_mastering.set(False)
            self.normalize_audio.set(False)
            self.stem_rebalance.set(True)
        for widget in (self.cleanup_check, self.mastering_check, self.normalize_check):
            widget.state(["disabled"] if stem_mode else ["!disabled"])
        restoration_enabled = self.restoration_cleanup.get() and not stem_mode
        combo_state = "readonly" if restoration_enabled else "disabled"
        self.restoration_preset_combo.config(state=combo_state)
        self.hum_combo.config(state=combo_state)

        bandwidth_allowed = profile in {"restore", "advanced"}
        if not bandwidth_allowed:
            self.bandwidth_restore.set(False)
        self.bandwidth_check.state(["!disabled"] if bandwidth_allowed else ["disabled"])

        backend_allowed = advanced or stem_mode or self.stem_rebalance.get()
        self.backend_combo.config(state="readonly" if backend_allowed else "disabled")
        if not advanced and not stem_mode:
            self.backend_choice.set("roformer")

    def _refresh_state(self):
        state = self.backend.snapshot()
        self.status_text_var.set(state["status_text"])
        color = self.COLOR_MAP.get(state["status_color"], "#607D8B")
        self.status_label.configure(foreground=color)
        self.progress.configure(value=state["progress"])
        self.progress_value_var.set(f"{state['progress']:.0f}%")
        self.roformer_var.set(state["resolved_roformer_model"])
        self.original_name_var.set(state["original_name"])
        self.enhanced_name_var.set(state["enhanced_name"])
        self.session_dir_var.set(state["session_dir"] or "No active session")
        self.export_format.set(state["current_export_format"])

        self.download_button.config(state=("disabled" if state["processing"] else "normal"))
        self.import_path_var.set(self.import_path_var.get())
        self.apply_button.config(state=("disabled" if state["processing"] or not state["has_original"] else "normal"))
        self.undo_button.config(state=("normal" if state["can_undo"] else "disabled"))
        self.revert_button.config(state=("normal" if state["can_revert"] else "disabled"))
        self.export_button.config(state=("disabled" if state["processing"] or not state["has_original"] else "normal"))
        self._refresh_player_panel("original", state["has_original"], state["original_name"], active=(self.playback_kind == "original"))
        self._refresh_player_panel("enhanced", state["has_enhanced"], state["enhanced_name"], active=(self.playback_kind == "enhanced"))

        details = state["last_details"] or "No backend details."
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", details)
        self.details_text.config(state="disabled")

        current_original = self.backend.source_file
        current_enhanced = self.backend.current_file if state["has_enhanced"] else None
        self._invalidate_cache_if_needed("original", current_original)
        self._invalidate_cache_if_needed("enhanced", current_enhanced)

        self.root.after(1000, self._refresh_state)

    def _refresh_player_panel(self, kind, available, display_name, active=False):
        widgets = self.player_widgets[kind]
        widgets["name_var"].set(display_name if available else widgets["fallback_label"])

        play_state = "normal" if available and not self.backend.processing else "disabled"
        widgets["play_button"].config(state=play_state)
        widgets["pause_button"].config(state=("normal" if active and self.is_playing else "disabled"))
        widgets["stop_button"].config(state=("normal" if active and self.is_playing else "disabled"))

        if not available:
            widgets["status_var"].set("No audio ready yet")
            widgets["time_var"].set("0:00 / 0:00")
            widgets["seek_var"].set(0.0)
        elif active:
            widgets["status_var"].set("Paused" if self.is_paused else "Playing")
        else:
            widgets["status_var"].set("Ready to compare")

    def _invalidate_cache_if_needed(self, kind, source_path):
        cached = self.playback_cache.get(kind)
        if not cached:
            return
        if not source_path or cached["source_path"] != source_path:
            self.playback_cache.pop(kind, None)
            if self.playback_kind == kind:
                self.stop_audio()
            return
        try:
            if os.path.getmtime(source_path) != cached["mtime"]:
                self.playback_cache.pop(kind, None)
        except FileNotFoundError:
            self.playback_cache.pop(kind, None)

    def start_download(self):
        url = self.url_entry.get().strip()
        browser = self.browser_combo.get().strip().lower() or "chrome"
        try:
            self.backend.run_job(self.backend.download_audio, url, self.download_mode.get(), browser)
        except Exception as exc:
            messagebox.showerror("Download", str(exc))

    def import_file(self):
        path = filedialog.askopenfilename(
            title="Import audio file",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.webm *.opus *.ogg"), ("All files", "*.*")],
        )
        if not path:
            return
        self.import_path_var.set(Path(path).name)
        temp_dir = Path(tempfile.mkdtemp(prefix="restoration_tk_import_"))
        temp_copy = temp_dir / Path(path).name
        shutil.copy2(path, temp_copy)
        try:
            self.backend.run_job(self.backend.import_audio_from_path, str(temp_copy))
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            messagebox.showerror("Import", str(exc))

    def collect_options(self):
        return {
            "profile": self.selected_profile.get(),
            "restoration_cleanup": self.restoration_cleanup.get(),
            "restoration_preset": self._restoration_preset_key_from_label(self.restoration_preset.get()),
            "hum_frequency": self._normalize_hum_label(self.hum_frequency.get()).split()[0],
            "clarity_mastering": self.clarity_mastering.get(),
            "normalize_audio": self.normalize_audio.get(),
            "stem_rebalance": self.stem_rebalance.get(),
            "bandwidth_restore": self.bandwidth_restore.get(),
            "backend": self.backend_choice.get(),
            "bass_boost": int(self.bass_boost.get()),
            "treble_boost": int(self.treble_boost.get()),
            "volume_boost": int(self.volume_boost.get()),
        }

    def apply_profile(self):
        try:
            self.backend.run_job(self.backend.enhance_current, self.collect_options())
        except Exception as exc:
            messagebox.showerror("Apply Profile", str(exc))

    def undo_last(self):
        try:
            self.backend.undo_last()
        except Exception as exc:
            messagebox.showinfo("Undo", str(exc))

    def revert_to_original(self):
        try:
            self.backend.revert_to_original()
        except Exception as exc:
            messagebox.showinfo("Revert", str(exc))

    def export_current(self):
        fmt = self.export_format.get()
        default_name = "restored_track." + fmt
        path = filedialog.asksaveasfilename(
            title="Export current result",
            defaultextension=f".{fmt}",
            initialfile=default_name,
            filetypes=[(fmt.upper(), f"*.{fmt}"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            exported = self.backend.export_current(fmt)
            shutil.copy2(exported, path)
            messagebox.showinfo("Export", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def _ensure_playable_file(self, kind):
        source_path = self.backend.source_file if kind == "original" else self.backend.current_file
        if not source_path or not os.path.exists(source_path):
            raise RuntimeError("No audio is available for playback.")

        session_dir = Path(self.backend.current_session_dir or ROOT)
        cache_dir = session_dir / "playback_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{kind}.wav"
        mtime = os.path.getmtime(source_path)

        cached = self.playback_cache.get(kind)
        if cached and cached["source_path"] == source_path and cached["mtime"] == mtime and cache_path.exists():
            return str(cache_path)

        self.backend._convert_to_work_wav(source_path, str(cache_path))
        self.playback_cache[kind] = {"source_path": source_path, "mtime": mtime, "cache_path": str(cache_path)}
        return str(cache_path)

    def _create_seek_segment(self, full_wav, kind, start_seconds):
        if start_seconds <= 0:
            return full_wav
        session_dir = Path(self.backend.current_session_dir or ROOT)
        cache_dir = session_dir / "playback_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        previous_segment = self.seek_segment_paths.get(kind)
        if previous_segment and os.path.exists(previous_segment):
            try:
                os.remove(previous_segment)
            except OSError:
                pass
        fd, segment_temp = tempfile.mkstemp(prefix=f"{kind}_seek_", suffix=".wav", dir=cache_dir)
        os.close(fd)
        segment_path = Path(segment_temp)
        with wave.open(full_wav, "rb") as source, wave.open(str(segment_path), "wb") as output:
            output.setparams(source.getparams())
            frame_rate = source.getframerate() or 1
            start_frame = min(int(start_seconds * frame_rate), source.getnframes())
            source.setpos(start_frame)
            while True:
                chunk = source.readframes(65536)
                if not chunk:
                    break
                output.writeframes(chunk)
        self.seek_segment_paths[kind] = str(segment_path)
        return str(segment_path)

    def _load_audio_at_position(self, kind, start_seconds=0.0):
        full_wav = self._ensure_playable_file(kind)
        start_seconds = max(0.0, min(start_seconds, self._wav_length(full_wav)))
        play_file = self._create_seek_segment(full_wav, kind, start_seconds)
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        pygame.mixer.music.load(play_file)
        pygame.mixer.music.play()
        self.playback_kind = kind
        self.playback_source = full_wav
        self.song_length = self._wav_length(full_wav)
        self.playback_base_position = start_seconds
        self.playback_start_time = time.time()
        self.paused_position = start_seconds
        self.is_playing = True
        self.is_paused = False
        for player_kind, widgets in self.player_widgets.items():
            if player_kind != kind:
                widgets["status_var"].set("Ready to compare")
                widgets["time_var"].set("0:00 / 0:00")
                widgets["seek_var"].set(0.0)
        self.player_widgets[kind]["status_var"].set("Playing")
        self.player_widgets[kind]["seek_var"].set((start_seconds / self.song_length) * 100.0 if self.song_length else 0.0)

    def play_audio(self, kind, start_seconds=None):
        try:
            if self.playback_kind == kind and self.is_paused and start_seconds is None:
                pygame.mixer.music.unpause()
                self.is_paused = False
                self.playback_start_time = time.time()
                self.player_widgets[kind]["status_var"].set("Playing")
                return
            self._load_audio_at_position(kind, start_seconds or 0.0)
        except Exception as exc:
            messagebox.showerror("Playback", str(exc))

    def _wav_length(self, wav_path):
        with wave.open(wav_path, "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return frames / float(rate or 1)

    def pause_audio(self, kind=None):
        if not self.is_playing or (kind and kind != self.playback_kind):
            return
        if self.is_paused:
            self.playback_base_position = self.paused_position
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.playback_start_time = time.time()
            self.player_widgets[self.playback_kind]["status_var"].set("Playing")
            return
        self.paused_position = self.get_playback_seconds()
        pygame.mixer.music.pause()
        self.is_paused = True
        self.player_widgets[self.playback_kind]["status_var"].set("Paused")

    def stop_audio(self, kind=None):
        if kind and self.playback_kind and kind != self.playback_kind:
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass
        self.is_playing = False
        self.is_paused = False
        self.playback_base_position = 0.0
        self.playback_start_time = None
        self.paused_position = 0.0
        self.song_length = 0.0
        for widgets in self.player_widgets.values():
            widgets["status_var"].set("Ready to compare")
            widgets["time_var"].set("0:00 / 0:00")
            widgets["seek_var"].set(0.0)
        self._cleanup_seek_segments()

    def get_playback_seconds(self):
        if self.is_playing:
            if self.is_paused:
                return self.paused_position
            if self.playback_start_time is not None:
                elapsed = max(0.0, time.time() - self.playback_start_time)
                return min(self.song_length, self.playback_base_position + elapsed)
        return self.paused_position

    def _update_playback_position(self):
        if self.is_playing:
            current = self.get_playback_seconds()
            if not pygame.mixer.music.get_busy() and not self.is_paused:
                self.stop_audio()
            else:
                widgets = self.player_widgets.get(self.playback_kind)
                if widgets:
                    widgets["time_var"].set(f"{self._format_time(current)} / {self._format_time(self.song_length)}")
                if self.song_length > 0:
                    value = (current / self.song_length) * 100.0
                else:
                    value = 0
                if widgets and self.seek_dragging_kind != self.playback_kind:
                    widgets["seek_var"].set(value)
        self.root.after(250, self._update_playback_position)

    def _begin_seek(self, kind):
        self.seek_dragging_kind = kind

    def _commit_seek(self, kind):
        widgets = self.player_widgets[kind]
        self.seek_dragging_kind = None
        source_path = self.backend.source_file if kind == "original" else self.backend.current_file
        if not source_path or not os.path.exists(source_path):
            widgets["seek_var"].set(0.0)
            return
        try:
            full_wav = self._ensure_playable_file(kind)
            duration = self._wav_length(full_wav)
            percent = max(0.0, min(100.0, float(widgets["seek_var"].get())))
            target_seconds = duration * (percent / 100.0)
            self.play_audio(kind, start_seconds=target_seconds)
        except Exception as exc:
            widgets["status_var"].set("Seek failed")
            messagebox.showerror("Playback", str(exc))

    def _cleanup_seek_segments(self):
        for kind, segment_path in list(self.seek_segment_paths.items()):
            if not segment_path or not os.path.exists(segment_path):
                self.seek_segment_paths.pop(kind, None)
                continue
            try:
                os.remove(segment_path)
                self.seek_segment_paths.pop(kind, None)
            except OSError:
                pass

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def toggle_details(self):
        if self.show_details.get():
            self.show_details.set(False)
            self.details_frame.pack_forget()
            self.toggle_details_button.config(text="Show details")
        else:
            self.show_details.set(True)
            self.details_frame.pack(fill="both", expand=True, pady=(10, 0))
            self.toggle_details_button.config(text="Hide details")


def main():
    root = tk.Tk()
    app = RestorationStudioTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

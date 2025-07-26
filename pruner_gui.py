# FILE: pruner_gui.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import sys
import io
from pathlib import Path
import logging  #

# Import the enhanced pruner logic from the other file
from model_pruner import ModelPruner, logger

class PrunerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced AI Model Pruner")
        self.geometry("800x650")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # --- UI Elements ---
        self.create_widgets()

    def create_widgets(self):
        # Frame for file paths
        path_frame = ctk.CTkFrame(self)
        path_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        path_frame.grid_columnconfigure(1, weight=1)

        self.input_path_label = ctk.CTkLabel(path_frame, text="Input Model Path:")
        self.input_path_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.input_path_entry = ctk.CTkEntry(path_frame, placeholder_text="Select model folder...")
        self.input_path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.input_browse_button = ctk.CTkButton(path_frame, text="Browse...", command=self.browse_input)
        self.input_browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.output_path_label = ctk.CTkLabel(path_frame, text="Output Model Path:")
        self.output_path_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.output_path_entry = ctk.CTkEntry(path_frame, placeholder_text="Select output folder...")
        self.output_path_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.output_browse_button = ctk.CTkButton(path_frame, text="Browse...", command=self.browse_output)
        self.output_browse_button.grid(row=1, column=2, padx=10, pady=10)

        # Frame for settings
        settings_frame = ctk.CTkFrame(self)
        settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)

        self.strategy_label = ctk.CTkLabel(settings_frame, text="Pruning Strategy:")
        self.strategy_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.strategy_menu = ctk.CTkOptionMenu(settings_frame, values=['comprehensive', 'gradual', 'structured', 'magnitude', 'distillation'])
        self.strategy_menu.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.reduction_label = ctk.CTkLabel(settings_frame, text="Target Reduction: 75%")
        self.reduction_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.reduction_slider = ctk.CTkSlider(settings_frame, from_=0.1, to=0.9, command=self.update_reduction_label)
        self.reduction_slider.set(0.75)
        self.reduction_slider.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.validate_check = ctk.CTkCheckBox(settings_frame, text="Validate Pruned Model")
        self.validate_check.grid(row=0, column=2, padx=20, pady=10)
        self.validate_check.select()

        # Frame for output log
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_textbox = ctk.CTkTextbox(log_frame, wrap="word")
        self.log_textbox.grid(row=0, column=0, sticky="nsew")
        self.log_textbox.insert("end", "Welcome to the AI Model Pruner!\n\n")
        self.log_textbox.insert("end", "1. Select the input model folder.\n")
        self.log_textbox.insert("end", "2. Select the output folder for the pruned model.\n")
        self.log_textbox.insert("end", "3. Choose a strategy and reduction target.\n")
        self.log_textbox.insert("end", "4. Click 'Start Pruning'.\n")

        # Progress bar and start button
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        self.progress_bar.set(0)

        self.start_button = ctk.CTkButton(self, text="Start Pruning", height=40, command=self.start_pruning_thread)
        self.start_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

    def browse_input(self):
        path = filedialog.askdirectory(title="Select Input Model Folder")
        if path:
            self.input_path_entry.delete(0, "end")
            self.input_path_entry.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_path_entry.delete(0, "end")
            self.output_path_entry.insert(0, path)

    def update_reduction_label(self, value):
        self.reduction_label.configure(text=f"Target Reduction: {int(value*100)}%")

    def log(self, message):
        """Safely appends a message to the log textbox from any thread."""
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.see("end")

    def start_pruning_thread(self):
        input_path = self.input_path_entry.get()
        output_path = self.output_path_entry.get()
        
        if not input_path or not output_path:
            messagebox.showerror("Error", "Input and Output paths are required.")
            return

        self.start_button.configure(state="disabled", text="Pruning in progress...")
        self.progress_bar.start()

        pruning_thread = threading.Thread(
            target=self.run_pruning_task,
            args=(input_path, output_path),
            daemon=True
        )
        pruning_thread.start()

    def run_pruning_task(self, input_path, output_path):
        """The actual pruning logic that runs in a background thread."""
        try:
            # Redirect logger output to the GUI
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            
            self.log_textbox.delete("1.0", "end")
            self.log("Starting pruning process...")

            # --- Get GUI settings ---
            strategy = self.strategy_menu.get()
            reduction = self.reduction_slider.get()
            validate = self.validate_check.get()

            # --- Initialize and run pruner ---
            pruner = ModelPruner(
                model_path=input_path,
                output_path=output_path,
                target_reduction=reduction
            )
            
            # This is a bit of a hack to stream logs, but effective for this app
            def update_log():
                log_contents = log_stream.getvalue()
                if log_contents:
                    self.log_textbox.insert("end", log_contents)
                    self.log_textbox.see("end")
                    log_stream.truncate(0)
                    log_stream.seek(0)
                if pruning_thread.is_alive():
                    self.after(500, update_log)

            pruning_thread = threading.current_thread()
            self.after(500, update_log)
            
            pruner.load_model()
            
            if strategy == 'magnitude':
                pruner.magnitude_based_pruning(sparsity_ratio=reduction)
            elif strategy == 'structured':
                pruner.structured_pruning(reduction_ratio=reduction)
            elif strategy == 'gradual':
                pruner.gradual_magnitude_pruning(final_sparsity=reduction, steps=5)
            elif strategy == 'distillation':
                pruner.knowledge_distillation_pruning(student_ratio=1-reduction, epochs=3)
            else: # comprehensive
                pruner.optimize_model_size()
            
            if validate:
                pruner.validate_model_output()
            
            pruner.save_pruned_model()
            report = pruner.generate_report()
            
            self.log("\n" + "="*20 + " PROCESS COMPLETE " + "="*20)
            self.log(report)
            
        except Exception as e:
            self.log(f"\n\n--- AN ERROR OCCURRED ---\n{e}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Pruning Failed", f"An error occurred: {e}")
        finally:
            # --- Reset GUI state ---
            self.progress_bar.stop()
            self.progress_bar.set(0)
            self.start_button.configure(state="normal", text="Start Pruning")
            logger.removeHandler(handler) # Clean up logger

if __name__ == "__main__":
    app = PrunerApp()
    app.mainloop()
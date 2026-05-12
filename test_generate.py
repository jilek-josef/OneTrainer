#!/usr/bin/env python3
"""
Quick test script to generate a single image using the DistillLoRA setup.
Run this after starting OneTrainer to generate a fresh test image.
"""
import sys
sys.path.insert(0, '/home/pc/KimiProjects/ZImage-Loli/OneTrainer')

# This is a placeholder - the actual generation requires the full
# OneTrainer infrastructure. Instead, just run:
#   ./start-ui.sh
# and check the debug images in debug/distill_lora/

print("Please run ./start-ui.sh and check debug/distill_lora/ for new images")
print("The latest student image will be at:")
print("  debug/distill_lora/student_step_XXXXXX.png")

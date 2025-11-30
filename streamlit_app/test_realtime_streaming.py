#!/usr/bin/env python3
"""
Quick test script to verify RealTimeLogStreamer functionality.
Run this to see real-time log streaming in action.
"""

import subprocess
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "streamlit_app"))

from realtime_log_streamer import RealTimeLogStreamer


def create_test_script():
    """Create a test training script that generates logs."""
    script = '''
import time
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

logger.info("Position 1/3")
logger.info("=== XGBoost Position 1 ===")

for trial in range(1, 6):
    score = 0.1 * trial
    logger.info(f"  Trial {trial}/5: Score={score:.4f}")
    if trial == 3:
        logger.info(f"    [BEST] NEW BEST! Score: {score:.4f}")
    time.sleep(0.5)

logger.info("[BEST] XGBoost complete. Test Score: 0.5000")

logger.info("Position 2/3")
logger.info("=== LightGBM Position 2 ===")

for trial in range(1, 4):
    score = 0.2 * trial
    logger.info(f"  Trial {trial}/4: Score={score:.4f}")
    time.sleep(0.3)

logger.info("[BEST] LightGBM complete. Test Score: 0.6000")
logger.info("Training completed successfully!")
'''
    script_path = Path("test_trainer.py")
    script_path.write_text(script)
    return script_path


def main():
    """Run the test."""
    print("[START] RealTimeLogStreamer Test")
    print("=" * 60)
    
    # Create test script
    script_path = create_test_script()
    print(f"[OK] Created test script: {script_path}")
    
    # Create log file path
    log_file = Path("test_training.log")
    print(f"[OK] Log file: {log_file}")
    
    # Start process
    print("\n[INIT] Starting subprocess...")
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    print(f"[OK] Process started (PID: {process.pid})")
    
    # Create streamer
    print("\n[INIT] Initializing RealTimeLogStreamer...")
    streamer = RealTimeLogStreamer(process, log_file, max_buffer_lines=100)
    streamer.start()
    print("[OK] Streamer started")
    
    # Monitor progress
    print("\n[MONITOR] Progress in real-time:\n")
    
    iteration = 0
    while streamer.is_process_running():
        iteration += 1
        
        # Get progress
        progress = streamer.get_progress()
        
        # Display progress
        print(f"\n[{iteration:02d}] Position: {progress.position}/{progress.total_positions} | "
              f"Model: {progress.model_name} | "
              f"Trial: {progress.trial}/{progress.total_trials} | "
              f"Score: {progress.score:.4f} | "
              f"Best: {'Y' if progress.is_best else 'N'} | "
              f"Elapsed: {progress.elapsed_time}")
        
        # Get logs
        logs = streamer.get_logs(num_lines=50)
        if logs:
            print("Last 3 logs:")
            for line in logs[-3:]:
                print(f"  > {line}")
        
        time.sleep(1)
    
    # Final stats
    print("\n" + "=" * 60)
    print("[DONE] Process completed")
    
    final_progress = streamer.get_progress()
    print(f"\nFinal Progress:")
    print(f"  Position: {final_progress.position}/{final_progress.total_positions}")
    print(f"  Model: {final_progress.model_name}")
    print(f"  Trial: {final_progress.trial}/{final_progress.total_trials}")
    print(f"  Score: {final_progress.score:.4f}")
    print(f"  Elapsed: {final_progress.elapsed_time}")
    
    # Get all logs
    all_logs = streamer.get_logs()
    print(f"\nTotal logs captured: {len(all_logs)}")
    
    # Cleanup
    streamer.stop()
    process.wait()
    
    print("\n[OK] Test completed successfully!")
    print(f"[OK] Logs saved to: {log_file}")
    
    # Show saved log file
    if log_file.exists():
        print(f"\nSaved log contents:")
        print("-" * 60)
        print(log_file.read_text())
    
    # Cleanup test files
    script_path.unlink()
    if log_file.exists():
        log_file.unlink()
    print("[OK] Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

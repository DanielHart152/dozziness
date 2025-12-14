#!/usr/bin/env python3
"""
Debug log viewer for the enhanced scoring system.
Provides real-time monitoring and filtering of debug logs.
"""

import os
import time
import argparse
from datetime import datetime

def tail_file(filename, lines=50):
    """Read last N lines from file."""
    try:
        with open(filename, 'r') as f:
            return f.readlines()[-lines:]
    except FileNotFoundError:
        return [f"Log file {filename} not found. Run the system first to generate logs.\n"]

def filter_logs(lines, filter_type=None):
    """Filter log lines by type."""
    if not filter_type:
        return lines
    
    filters = {
        'score': ['SCORE CALCULATION', 'Score components', 'Final score'],
        'perclos': ['PERCLOS:'],
        'blinks': ['BLINKS'],
        'penalty': ['PENALTY'],
        'alert': ['ALERT', 'Alert state', 'Rolling avg']
    }
    
    keywords = filters.get(filter_type.lower(), [filter_type])
    return [line for line in lines if any(keyword in line for keyword in keywords)]

def format_log_line(line):
    """Format log line for better readability."""
    if 'SCORE CALCULATION START' in line:
        return f"\n{'='*60}\n{line.strip()}\n{'='*60}"
    elif 'ALERT TRIGGERED' in line or 'ALERT CLEARED' in line:
        return f"🚨 {line.strip()}"
    elif 'PENALTY awarded' in line:
        return f"⚠️  {line.strip()}"
    elif 'Final score:' in line:
        return f"📊 {line.strip()}"
    else:
        return line.strip()

def main():
    parser = argparse.ArgumentParser(description='View debug logs from scoring system')
    parser.add_argument('--file', default='scorer_debug.log', help='Debug log file path')
    parser.add_argument('--lines', type=int, default=50, help='Number of lines to show')
    parser.add_argument('--filter', choices=['score', 'perclos', 'blinks', 'penalty', 'alert'], 
                       help='Filter logs by type')
    parser.add_argument('--follow', '-f', action='store_true', help='Follow log file (like tail -f)')
    parser.add_argument('--clear', action='store_true', help='Clear screen before showing logs')
    
    args = parser.parse_args()
    
    if args.clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"Debug Log Viewer - {args.file}")
    print(f"Filter: {args.filter or 'None'} | Lines: {args.lines} | Follow: {args.follow}")
    print("=" * 80)
    
    if args.follow:
        # Follow mode - continuously monitor file
        last_size = 0
        try:
            while True:
                if os.path.exists(args.file):
                    current_size = os.path.getsize(args.file)
                    if current_size > last_size:
                        # File has grown, read new content
                        with open(args.file, 'r') as f:
                            f.seek(last_size)
                            new_lines = f.readlines()
                            
                        filtered_lines = filter_logs(new_lines, args.filter)
                        for line in filtered_lines:
                            print(format_log_line(line))
                        
                        last_size = current_size
                
                time.sleep(0.5)  # Check every 500ms
                
        except KeyboardInterrupt:
            print("\nStopped following log file.")
    else:
        # Static mode - show last N lines
        lines = tail_file(args.file, args.lines)
        filtered_lines = filter_logs(lines, args.filter)
        
        for line in filtered_lines:
            print(format_log_line(line))
        
        print(f"\nShowing {len(filtered_lines)} lines (filtered from {len(lines)} total)")

if __name__ == "__main__":
    main()
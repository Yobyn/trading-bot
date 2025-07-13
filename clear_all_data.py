#!/usr/bin/env python3
"""
Comprehensive Data Cleanup Script for Trading Bot
Clears all data including SQLite databases, JSON files, logs, and position history
"""

import os
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from loguru import logger

def clear_sqlite_databases():
    """Clear all SQLite databases"""
    print("🗄️ Clearing SQLite databases...")
    
    # Performance monitor database
    performance_db = "performance_data/performance.db"
    if os.path.exists(performance_db):
        try:
            conn = sqlite3.connect(performance_db)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM portfolio_snapshots")
            cursor.execute("DELETE FROM performance_metrics")
            cursor.execute("VACUUM")
            conn.commit()
            conn.close()
            print(f"✅ Cleared performance database: {performance_db}")
        except Exception as e:
            print(f"❌ Error clearing performance database: {e}")
    
    # Audit trail database
    audit_db = "audit_trails/audit_trail.db"
    if os.path.exists(audit_db):
        try:
            conn = sqlite3.connect(audit_db)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM llm_interactions")
            cursor.execute("DELETE FROM trading_decisions")
            cursor.execute("DELETE FROM portfolio_snapshots")
            cursor.execute("VACUUM")
            conn.commit()
            conn.close()
            print(f"✅ Cleared audit trail database: {audit_db}")
        except Exception as e:
            print(f"❌ Error clearing audit trail database: {e}")
    
    # Risk manager database
    risk_db = "risk_data/risk_data.db"
    if os.path.exists(risk_db):
        try:
            conn = sqlite3.connect(risk_db)
            cursor = conn.cursor()
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                if table[0] != 'sqlite_sequence':  # Skip system table
                    cursor.execute(f"DELETE FROM {table[0]}")
            
            cursor.execute("VACUUM")
            conn.commit()
            conn.close()
            print(f"✅ Cleared risk database: {risk_db}")
        except Exception as e:
            print(f"❌ Error clearing risk database: {e}")

def clear_json_files():
    """Clear all JSON audit trail files"""
    print("📄 Clearing JSON audit trail files...")
    
    json_directories = [
        "audit_trails/llm_interactions",
        "audit_trails/trading_decisions",
        "audit_trails/portfolio_snapshots"
    ]
    
    total_files = 0
    for directory in json_directories:
        if os.path.exists(directory):
            try:
                files = [f for f in os.listdir(directory) if f.endswith('.json')]
                for file in files:
                    os.remove(os.path.join(directory, file))
                    total_files += 1
                print(f"✅ Cleared {len(files)} JSON files from {directory}")
            except Exception as e:
                print(f"❌ Error clearing JSON files from {directory}: {e}")
    
    print(f"📊 Total JSON files cleared: {total_files}")

def clear_position_history():
    """Clear position history files"""
    print("📈 Clearing position history files...")
    
    position_files = [
        "position_history.json",
        "coinbase_position_history.json"
    ]
    
    # Look for position history files in current directory and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file in position_files:
                filepath = os.path.join(root, file)
                try:
                    os.remove(filepath)
                    print(f"✅ Removed position history file: {filepath}")
                except Exception as e:
                    print(f"❌ Error removing {filepath}: {e}")

def clear_analysis_files():
    """Clear analysis files"""
    print("📊 Clearing analysis files...")
    
    analysis_dir = "analysis"
    if os.path.exists(analysis_dir):
        try:
            files = os.listdir(analysis_dir)
            for file in files:
                filepath = os.path.join(analysis_dir, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            print(f"✅ Cleared {len(files)} analysis files")
        except Exception as e:
            print(f"❌ Error clearing analysis files: {e}")

def clear_log_files():
    """Clear log files"""
    print("📝 Clearing log files...")
    
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        try:
            files = os.listdir(logs_dir)
            for file in files:
                filepath = os.path.join(logs_dir, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            print(f"✅ Cleared {len(files)} log files")
        except Exception as e:
            print(f"❌ Error clearing log files: {e}")

def clear_temporary_files():
    """Clear temporary and cache files"""
    print("🧹 Clearing temporary files...")
    
    temp_patterns = [
        "*.tmp",
        "*.cache",
        "__pycache__",
        "*.pyc",
        ".DS_Store"
    ]
    
    total_cleared = 0
    for root, dirs, files in os.walk("."):
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"✅ Removed __pycache__ directory: {pycache_path}")
                total_cleared += 1
            except Exception as e:
                print(f"❌ Error removing __pycache__ directory {pycache_path}: {e}")
        
        # Remove temporary files
        for file in files:
            if (file.endswith('.tmp') or file.endswith('.cache') or 
                file.endswith('.pyc') or file == '.DS_Store'):
                filepath = os.path.join(root, file)
                try:
                    os.remove(filepath)
                    total_cleared += 1
                except Exception as e:
                    print(f"❌ Error removing temporary file {filepath}: {e}")
    
    if total_cleared > 0:
        print(f"✅ Cleared {total_cleared} temporary files")

def verify_cleanup():
    """Verify that cleanup was successful"""
    print("🔍 Verifying cleanup...")
    
    # Check database record counts
    databases = [
        ("performance_data/performance.db", ["trades", "portfolio_snapshots", "performance_metrics"]),
        ("audit_trails/audit_trail.db", ["llm_interactions", "trading_decisions", "portfolio_snapshots"]),
        ("risk_data/risk_data.db", None)  # Will check all tables
    ]
    
    for db_path, tables in databases:
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                if tables is None:
                    # Get all table names
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [table[0] for table in cursor.fetchall() if table[0] != 'sqlite_sequence']
                
                all_empty = True
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        all_empty = False
                        print(f"⚠️ {db_path} table {table} still has {count} records")
                
                if all_empty:
                    print(f"✅ {db_path} - All tables empty")
                
                conn.close()
            except Exception as e:
                print(f"❌ Error verifying {db_path}: {e}")
    
    # Check JSON file counts
    json_dirs = [
        "audit_trails/llm_interactions",
        "audit_trails/trading_decisions", 
        "audit_trails/portfolio_snapshots"
    ]
    
    for directory in json_dirs:
        if os.path.exists(directory):
            json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
            if json_files:
                print(f"⚠️ {directory} still has {len(json_files)} JSON files")
            else:
                print(f"✅ {directory} - No JSON files")
    
    # Check for position history files
    position_found = False
    for root, dirs, files in os.walk("."):
        for file in files:
            if "position_history" in file and file.endswith('.json'):
                print(f"⚠️ Position history file still exists: {os.path.join(root, file)}")
                position_found = True
    
    if not position_found:
        print("✅ No position history files found")

def main():
    """Main cleanup function"""
    print("🧹 Starting comprehensive data cleanup...")
    print("=" * 50)
    
    # Confirm with user
    response = input("This will delete ALL trading bot data. Are you sure? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cleanup cancelled")
        return
    
    print(f"🕐 Cleanup started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Perform cleanup operations
    clear_sqlite_databases()
    print()
    
    clear_json_files()
    print()
    
    clear_position_history()
    print()
    
    clear_analysis_files()
    print()
    
    clear_log_files()
    print()
    
    clear_temporary_files()
    print()
    
    # Verify cleanup
    verify_cleanup()
    print()
    
    print("=" * 50)
    print("✅ Comprehensive data cleanup completed!")
    print(f"🕐 Cleanup finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📝 Summary:")
    print("   - SQLite databases cleared")
    print("   - JSON audit trail files removed")
    print("   - Position history files removed")
    print("   - Analysis files cleared")
    print("   - Log files cleared")
    print("   - Temporary files cleaned")
    print()
    print("🚀 The trading bot is now ready for fresh data!")

if __name__ == "__main__":
    main() 
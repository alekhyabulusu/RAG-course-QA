#!/usr/bin/env python
"""
Comprehensive verification script for RAG Course QA project setup.
Checks all dependencies, API keys, and system requirements.
"""

import sys
import os
import importlib
import subprocess
import json
import platform
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text: str, color: str = Colors.ENDC, bold: bool = False):
    """Print colored text to terminal"""
    if bold:
        print(f"{color}{Colors.BOLD}{text}{Colors.ENDC}")
    else:
        print(f"{color}{text}{Colors.ENDC}")

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print_colored(f"  {text}", Colors.HEADER, bold=True)
    print("=" * 60)

def print_success(text: str):
    """Print success message"""
    print_colored(f"✅ {text}", Colors.GREEN)

def print_error(text: str):
    """Print error message"""
    print_colored(f"❌ {text}", Colors.RED)

def print_warning(text: str):
    """Print warning message"""
    print_colored(f"⚠️  {text}", Colors.YELLOW)

def print_info(text: str):
    """Print info message"""
    print_colored(f"ℹ️  {text}", Colors.CYAN)

# =====================================================
# Package Checks
# =====================================================

def check_package(module_name: str, package_name: str = None, 
                  min_version: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and optionally verify version"""
    package_name = package_name or module_name
    
    try:
        module = importlib.import_module(module_name)
        
        # Try to get version
        version = "unknown"
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = str(getattr(module, attr))
                break
        
        # Check minimum version if specified
        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"{version} (requires >= {min_version})"
        
        return True, version
    
    except ImportError:
        return False, "Not installed"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_core_packages() -> Dict[str, Tuple[bool, str]]:
    """Check all core packages"""
    packages = [
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("langchain_openai", "LangChain OpenAI"),
        ("llama_index", "LlamaIndex"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("openai", "OpenAI"),
        ("chromadb", "ChromaDB"),
        ("pinecone", "Pinecone"),
        ("cohere", "Cohere"),
        ("sentence_transformers", "Sentence Transformers"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pypdf", "PyPDF"),
        ("dotenv", "python-dotenv"),
        ("tiktoken", "Tiktoken"),
    ]
    
    results = {}
    for module, name in packages:
        results[name] = check_package(module)
    
    return results

# =====================================================
# Environment and API Key Checks
# =====================================================

def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists and is properly formatted"""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if not env_path.exists():
        if example_path.exists():
            return False, ".env not found (but .env.example exists)"
        else:
            return False, "Neither .env nor .env.example found"
    
    # Check if .env is not empty
    if env_path.stat().st_size == 0:
        return False, ".env exists but is empty"
    
    # Try to load it
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Count configured variables
        with open(env_path, 'r') as f:
            lines = f.readlines()
            env_vars = [l for l in lines if '=' in l and not l.strip().startswith('#')]
        
        return True, f"Found with {len(env_vars)} variables"
    except Exception as e:
        return False, f"Error loading .env: {str(e)}"

def check_api_keys() -> Dict[str, Tuple[bool, str]]:
    """Check which API keys are configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    keys_to_check = {
        "OPENAI_API_KEY": ("OpenAI", "Required", "sk-"),
        "PINECONE_API_KEY": ("Pinecone", "Optional", ""),
        "HUGGINGFACE_API_KEY": ("Hugging Face", "Optional", "hf_"),
        "COHERE_API_KEY": ("Cohere", "Optional", ""),
    }
    
    results = {}
    for env_var, (name, requirement, prefix) in keys_to_check.items():
        key = os.getenv(env_var)
        
        if key:
            # Validate key format if prefix is specified
            if prefix and not key.startswith(prefix):
                results[f"{name} ({requirement})"] = (
                    False, 
                    f"Invalid format (should start with {prefix})"
                )
            else:
                # Show first 10 chars for verification
                key_preview = f"{key[:10]}..." if len(key) > 10 else "Set"
                results[f"{name} ({requirement})"] = (True, key_preview)
        else:
            results[f"{name} ({requirement})"] = (False, "Not configured")
    
    return results

def test_openai_connection() -> Tuple[bool, str]:
    """Test OpenAI API connection"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return False, "API key not set"
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Try to list models
        models = client.models.list()
        model_count = len(list(models))
        
        return True, f"Connected (Access to {model_count}+ models)"
    except Exception as e:
        error_msg = str(e)
        if "Invalid authentication" in error_msg:
            return False, "Invalid API key"
        elif "Rate limit" in error_msg:
            return True, "Connected (rate limited)"
        else:
            return False, f"Connection failed: {error_msg[:50]}"

# =====================================================
# Directory Structure Checks
# =====================================================

def check_directory_structure() -> Dict[str, Tuple[bool, str]]:
    """Check if required directories exist"""
    required_dirs = {
        "src": "Source code",
        "src/preprocessing": "Preprocessing module",
        "src/retrieval": "Retrieval module",
        "src/generation": "Generation module",
        "src/evaluation": "Evaluation module",
        "src/api": "API module",
        "src/ui": "UI module",
        "data": "Data directory",
        "data/raw": "Raw data",
        "data/processed": "Processed data",
        "data/embeddings": "Embeddings storage",
        "data/chroma": "ChromaDB storage",
        "tests": "Test files",
        "notebooks": "Jupyter notebooks",
        "scripts": "Utility scripts",
        "configs": "Configuration files",
        "logs": "Log files",
    }
    
    results = {}
    for dir_path, description in required_dirs.items():
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            # Count files in directory
            file_count = len(list(path.glob("*")))
            results[description] = (True, f"{file_count} files")
        else:
            results[description] = (False, "Not found")
    
    return results

def check_required_files() -> Dict[str, Tuple[bool, str]]:
    """Check if required files exist"""
    required_files = {
        "requirements.txt": "Dependencies",
        ".env": "Environment variables",
        ".env.example": "Environment template",
        ".gitignore": "Git ignore rules",
        "README.md": "Documentation",
        "src/config.py": "Configuration module",
    }
    
    results = {}
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            if size > 1024:
                size_str = f"{size // 1024} KB"
            else:
                size_str = f"{size} bytes"
            results[description] = (True, size_str)
        else:
            results[description] = (False, "Not found")
    
    return results

# =====================================================
# Configuration Tests
# =====================================================

def test_configuration() -> Tuple[bool, str]:
    """Test if configuration module loads properly"""
    try:
        from src.config import config, validate_environment
        
        # Check if configuration loads
        if validate_environment():
            return True, f"Loaded ({config.app.app_env} environment)"
        else:
            return False, "Configuration invalid"
    
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"

def test_vector_databases() -> Dict[str, Tuple[bool, str]]:
    """Test vector database connections"""
    results = {}
    
    # Test ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./data/chroma_test")
        collection = client.get_or_create_collection("test")
        results["ChromaDB (Local)"] = (True, "Working")
        
        # Clean up test
        client.delete_collection("test")
    except Exception as e:
        results["ChromaDB (Local)"] = (False, f"Error: {str(e)[:30]}")
    
    # Test Pinecone (if API key exists)
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_key)
            indexes = pc.list_indexes()
            results["Pinecone (Cloud)"] = (True, f"{len(indexes)} indexes")
        except Exception as e:
            results["Pinecone (Cloud)"] = (False, f"Error: {str(e)[:30]}")
    else:
        results["Pinecone (Cloud)"] = (False, "API key not set")
    
    return results

# =====================================================
# Performance Checks
# =====================================================

def check_gpu_availability() -> Tuple[bool, str]:
    """Check if GPU is available for PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{gpu_name} ({memory:.1f} GB)"
        else:
            return False, "No CUDA GPU available"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_memory_usage() -> Dict[str, str]:
    """Check current memory usage"""
    try:
        import psutil
        
        # Get memory info
        virtual_memory = psutil.virtual_memory()
        
        return {
            "Total RAM": f"{virtual_memory.total / 1e9:.1f} GB",
            "Available": f"{virtual_memory.available / 1e9:.1f} GB",
            "Used": f"{virtual_memory.percent:.1f}%",
            "Recommended": "✅ Sufficient" if virtual_memory.total > 8e9 else "⚠️ <8GB may be slow",
        }
    except ImportError:
        return {"Status": "psutil not installed"}

# =====================================================
# Report Generation
# =====================================================

def generate_report(results: Dict[str, any], save_to_file: bool = False) -> str:
    """Generate a comprehensive verification report"""
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report.append(f"RAG Course QA - Setup Verification Report")
    report.append(f"Generated: {timestamp}")
    report.append("=" * 60)
    
    # Add all results
    for section, data in results.items():
        report.append(f"\n{section}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, tuple):
                    status = "✅" if value[0] else "❌"
                    report.append(f"  {status} {key}: {value[1]}")
                else:
                    report.append(f"  {key}: {value}")
        elif isinstance(data, tuple):
            status = "✅" if data[0] else "❌"
            report.append(f"  {status} {data[1]}")
        else:
            report.append(f"  {data}")
    
    report_text = "\n".join(report)
    
    if save_to_file:
        report_path = Path("logs") / f"verification_report_{timestamp.replace(':', '-')}.txt"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)
        print_info(f"Report saved to: {report_path}")
    
    return report_text

# =====================================================
# Main Verification Function
# =====================================================

def run_verification(verbose: bool = True, save_report: bool = False):
    """Run all verification checks"""
    
    print_colored("\n🔍 RAG COURSE QA - SETUP VERIFICATION", Colors.HEADER, bold=True)
    print_colored("=" * 60, Colors.HEADER)
    
    all_results = {}
    total_checks = 0
    passed_checks = 0
    critical_failures = []
    
    # 1. Package Checks
    print_header("Required Packages")
    
    packages = check_core_packages()
    for package, (installed, version) in packages.items():
        if installed:
            print_success(f"{package}: {version}")
            passed_checks += 1
        else:
            print_error(f"{package}: {version}")
            if package in ["LangChain", "OpenAI", "ChromaDB"]:
                critical_failures.append(f"{package} not installed")
        total_checks += 1
    
    all_results["Packages"] = packages
    
    # 3. Environment and API Keys
    print_header("Environment Configuration")
    
    env_check = check_env_file()
    if env_check[0]:
        print_success(f".env file: {env_check[1]}")
        passed_checks += 1
    else:
        print_error(f".env file: {env_check[1]}")
        critical_failures.append(".env file not configured")
    total_checks += 1
    
    api_keys = check_api_keys()
    for service, (configured, status) in api_keys.items():
        if configured:
            print_success(f"{service}: {status}")
            passed_checks += 1
        else:
            if "Required" in service:
                print_error(f"{service}: {status}")
                critical_failures.append(f"{service} not configured")
            else:
                print_warning(f"{service}: {status}")
        total_checks += 1
    
    all_results["Environment"] = {
        ".env file": env_check,
        **api_keys
    }
    
    # 4. API Connection Tests
    print_header("API Connection Tests")
    
    openai_test = test_openai_connection()
    if openai_test[0]:
        print_success(f"OpenAI API: {openai_test[1]}")
        passed_checks += 1
    else:
        print_error(f"OpenAI API: {openai_test[1]}")
        if "API key not set" not in openai_test[1]:
            critical_failures.append("OpenAI API connection failed")
    total_checks += 1
    
    all_results["API Tests"] = {"OpenAI": openai_test}
    
    # 5. Directory Structure
    print_header("Project Structure")
    
    directories = check_directory_structure()
    missing_dirs = []
    for dir_name, (exists, status) in directories.items():
        if exists:
            if verbose:
                print_success(f"{dir_name}: {status}")
            passed_checks += 1
        else:
            print_error(f"{dir_name}: {status}")
            missing_dirs.append(dir_name)
        total_checks += 1
    
    if not verbose and not missing_dirs:
        print_success("All directories present")
    
    all_results["Directories"] = directories
    
    # 6. Required Files
    files = check_required_files()
    for file_name, (exists, status) in files.items():
        if exists:
            if verbose:
                print_success(f"{file_name}: {status}")
            passed_checks += 1
        else:
            print_error(f"{file_name}: {status}")
        total_checks += 1
    
    all_results["Files"] = files
    
    # 7. Configuration Test
    print_header("Configuration Module")
    
    config_test = test_configuration()
    if config_test[0]:
        print_success(f"Configuration: {config_test[1]}")
        passed_checks += 1
    else:
        print_error(f"Configuration: {config_test[1]}")
        critical_failures.append("Configuration module failed")
    total_checks += 1
    
    all_results["Configuration"] = config_test
    
    # 8. Vector Databases
    print_header("Vector Databases")
    
    vector_dbs = test_vector_databases()
    for db_name, (working, status) in vector_dbs.items():
        if working:
            print_success(f"{db_name}: {status}")
            passed_checks += 1
        else:
            if "Local" in db_name:
                print_error(f"{db_name}: {status}")
            else:
                print_warning(f"{db_name}: {status}")
        total_checks += 1
    
    all_results["Vector DBs"] = vector_dbs
    
    # 9. Performance Checks
    print_header("Performance & Resources")
    
    gpu_check = check_gpu_availability()
    if gpu_check[0]:
        print_success(f"GPU: {gpu_check[1]}")
    else:
        print_info(f"GPU: {gpu_check[1]}")
    
    memory_info = check_memory_usage()
    for key, value in memory_info.items():
        if "✅" in value:
            print_success(f"{key}: {value}")
        elif "⚠️" in value:
            print_warning(f"{key}: {value}")
        else:
            print_info(f"{key}: {value}")
    
    all_results["Performance"] = {
        "GPU": gpu_check,
        **memory_info
    }
    
    # Summary
    print_header("Verification Summary")
    
    success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\nTotal Checks: {total_checks}")
    print_success(f"Passed: {passed_checks}")
    print_error(f"Failed: {total_checks - passed_checks}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if critical_failures:
        print_colored("\n⚠️ Critical Issues Found:", Colors.RED, bold=True)
        for issue in critical_failures:
            print_error(issue)
        
        print_colored("\n📋 Next Steps:", Colors.YELLOW, bold=True)
        if ".env" in str(critical_failures):
            print("  1. Copy .env.example to .env and add your API keys")
        if "not installed" in str(critical_failures):
            print("  2. Run: pip install -r requirements.txt")
        if "OpenAI" in str(critical_failures) and "API key" not in str(critical_failures):
            print("  3. Check your OpenAI API key is valid")
    else:
        print_colored("\n🎉 All critical checks passed!", Colors.GREEN, bold=True)
        print_colored("Your environment is ready for the RAG project!", Colors.GREEN)
        
        print_colored("\n🚀 Quick Start:", Colors.CYAN, bold=True)
        print("  1. Process documents: python src/preprocessing/process_documents.py")
        print("  2. Start API: uvicorn src.api.main:app --reload")
        print("  3. Launch UI: streamlit run src/ui/app.py")
    
    # Save report if requested
    if save_report:
        report = generate_report(all_results, save_to_file=True)
    
    return success_rate >= 80  # Return True if 80% or more checks pass

# =====================================================
# Command Line Interface
# =====================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify RAG Course QA project setup")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed output")
    parser.add_argument("--save-report", "-s", action="store_true",
                       help="Save verification report to file")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run quick verification only")
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            # Quick check - just the essentials
            print_colored("\n🚀 Running quick verification...", Colors.CYAN, bold=True)
            
            # Check .env
            env_ok, env_msg = check_env_file()
            print(f".env: {'✅' if env_ok else '❌'} {env_msg}")
            
            # Check OpenAI
            api_keys = check_api_keys()
            openai_ok = api_keys.get("OpenAI (Required)", (False, ""))[0]
            print(f"OpenAI Key: {'✅' if openai_ok else '❌'}")
            
            # Check core packages
            try:
                import langchain
                import openai
                import chromadb
                print("Core Packages: ✅")
            except ImportError:
                print("Core Packages: ❌ Run: pip install -r requirements.txt")
            
            sys.exit(0 if env_ok else 1)
        
        # Run full verification
        success = run_verification(verbose=args.verbose, save_report=args.save_report)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print_colored("\n\n⚠️ Verification interrupted by user", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n\n❌ Unexpected error: {str(e)}", Colors.RED)
        sys.exit(1)
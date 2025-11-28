#!/bin/bash

# Deployment Script for Lottery Prediction System
# This script handles deployment across different environments

set -e  # Exit on any error

# Configuration
APP_NAME="lottery-prediction-system"
VERSION=$(cat streamlit_app/configs/constants.py | grep "APP_VERSION" | cut -d'"' -f2)
DEFAULT_ENV="production"
DEFAULT_PORT="8501"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Lottery Prediction System Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
  install     Install dependencies and setup environment
  start       Start the application
  stop        Stop the application
  restart     Restart the application
  status      Check application status
  backup      Create backup of data
  restore     Restore from backup
  update      Update application
  logs        Show application logs
  clean       Clean up temporary files

Options:
  -e, --env ENV       Environment (development|staging|production) [default: production]
  -p, --port PORT     Port number [default: 8501]
  -h, --help          Show this help message
  -v, --verbose       Verbose output
  --no-backup         Skip backup during update
  --force             Force operation without confirmation

Examples:
  $0 install                          # Install for production
  $0 -e development start             # Start in development mode
  $0 -p 8502 start                   # Start on custom port
  $0 backup                          # Create backup
  $0 update --no-backup              # Update without backup

EOF
}

# Parse command line arguments
ENVIRONMENT=$DEFAULT_ENV
PORT=$DEFAULT_PORT
VERBOSE=false
NO_BACKUP=false
FORCE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        install|start|stop|restart|status|backup|restore|update|logs|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production|testing)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

# Set environment variables
export LOTTERY_ENV=$ENVIRONMENT
export LOTTERY_PORT=$PORT

# Function to check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Using Python $python_version"
}

# Function to check if pip is available
check_pip() {
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        exit 1
    fi
}

# Function to create virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    log_success "Virtual environment activated"
}

# Function to install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    if [[ $VERBOSE == true ]]; then
        pip3 install -r requirements.txt
    else
        pip3 install -r requirements.txt --quiet
    fi
    
    log_success "Dependencies installed"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    directories=(
        "data"
        "logs"
        "cache"
        "exports"
        "models"
        "temp"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        if [[ $VERBOSE == true ]]; then
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directories created"
}

# Function to initialize database
init_database() {
    log_info "Initializing database..."
    
    # Run database initialization script if it exists
    if [[ -f "scripts/init_db.py" ]]; then
        python3 scripts/init_db.py
        log_success "Database initialized"
    else
        log_warning "Database initialization script not found"
    fi
}

# Function to start the application
start_app() {
    log_info "Starting $APP_NAME on port $PORT in $ENVIRONMENT mode..."
    
    # Check if already running
    if pgrep -f "streamlit run" > /dev/null; then
        log_warning "Application may already be running"
        if [[ $FORCE != true ]]; then
            log_error "Use --force to override or stop the application first"
            exit 1
        fi
    fi
    
    # Set additional environment variables
    export STREAMLIT_SERVER_PORT=$PORT
    export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
    
    if [[ $ENVIRONMENT == "production" ]]; then
        export STREAMLIT_SERVER_HEADLESS=true
        export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    fi
    
    # Start the application
    if [[ $ENVIRONMENT == "development" ]]; then
        streamlit run streamlit_app/app.py --server.port=$PORT
    else
        nohup streamlit run streamlit_app/app.py --server.port=$PORT > logs/app.log 2>&1 &
        echo $! > .app.pid
        log_success "Application started (PID: $(cat .app.pid))"
    fi
}

# Function to stop the application
stop_app() {
    log_info "Stopping $APP_NAME..."
    
    if [[ -f ".app.pid" ]]; then
        pid=$(cat .app.pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm .app.pid
            log_success "Application stopped"
        else
            log_warning "Process not found, cleaning up PID file"
            rm .app.pid
        fi
    else
        # Try to find and kill streamlit processes
        pkill -f "streamlit run" && log_success "Application stopped" || log_warning "No running application found"
    fi
}

# Function to check application status
check_status() {
    log_info "Checking application status..."
    
    if [[ -f ".app.pid" ]]; then
        pid=$(cat .app.pid)
        if kill -0 "$pid" 2>/dev/null; then
            log_success "Application is running (PID: $pid)"
            
            # Check if port is responding
            if command -v curl &> /dev/null; then
                if curl -s "http://localhost:$PORT" > /dev/null; then
                    log_success "Application is responding on port $PORT"
                else
                    log_warning "Application is running but not responding on port $PORT"
                fi
            fi
        else
            log_warning "PID file exists but process is not running"
            rm .app.pid
        fi
    else
        if pgrep -f "streamlit run" > /dev/null; then
            log_warning "Application may be running without PID file"
        else
            log_info "Application is not running"
        fi
    fi
}

# Function to create backup
create_backup() {
    log_info "Creating backup..."
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_dir="backups/backup_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Backup data directory
    if [[ -d "data" ]]; then
        cp -r data "$backup_dir/"
        log_info "Data backed up"
    fi
    
    # Backup configuration
    if [[ -d "streamlit_app/configs" ]]; then
        cp -r streamlit_app/configs "$backup_dir/"
        log_info "Configuration backed up"
    fi
    
    # Backup models
    if [[ -d "models" ]]; then
        cp -r models "$backup_dir/"
        log_info "Models backed up"
    fi
    
    # Create backup manifest
    cat > "$backup_dir/manifest.txt" << EOF
Backup created: $(date)
Environment: $ENVIRONMENT
Version: $VERSION
App PID: $(cat .app.pid 2>/dev/null || echo "Not running")
EOF
    
    log_success "Backup created: $backup_dir"
}

# Function to restore from backup
restore_backup() {
    log_info "Available backups:"
    ls -la backups/ | grep "backup_" | awk '{print $9}' | sort -r
    
    echo -n "Enter backup directory name (or 'latest' for most recent): "
    read backup_name
    
    if [[ "$backup_name" == "latest" ]]; then
        backup_name=$(ls backups/ | grep "backup_" | sort -r | head -n1)
    fi
    
    backup_path="backups/$backup_name"
    
    if [[ ! -d "$backup_path" ]]; then
        log_error "Backup not found: $backup_path"
        exit 1
    fi
    
    if [[ $FORCE != true ]]; then
        echo -n "This will overwrite current data. Continue? (y/N): "
        read confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            log_info "Restore cancelled"
            exit 0
        fi
    fi
    
    log_info "Restoring from $backup_path..."
    
    # Stop application first
    stop_app
    
    # Restore data
    if [[ -d "$backup_path/data" ]]; then
        rm -rf data
        cp -r "$backup_path/data" ./
        log_info "Data restored"
    fi
    
    # Restore configuration
    if [[ -d "$backup_path/configs" ]]; then
        rm -rf streamlit_app/configs
        cp -r "$backup_path/configs" streamlit_app/
        log_info "Configuration restored"
    fi
    
    # Restore models
    if [[ -d "$backup_path/models" ]]; then
        rm -rf models
        cp -r "$backup_path/models" ./
        log_info "Models restored"
    fi
    
    log_success "Restore completed"
}

# Function to update application
update_app() {
    log_info "Updating $APP_NAME..."
    
    # Create backup unless disabled
    if [[ $NO_BACKUP != true ]]; then
        create_backup
    fi
    
    # Pull latest code (if using git)
    if [[ -d ".git" ]]; then
        git pull origin main
        log_info "Code updated"
    fi
    
    # Update dependencies
    install_dependencies
    
    # Restart application
    stop_app
    sleep 2
    start_app
    
    log_success "Update completed"
}

# Function to show logs
show_logs() {
    log_info "Application logs:"
    
    if [[ -f "logs/app.log" ]]; then
        tail -f logs/app.log
    else
        log_warning "No log file found"
    fi
}

# Function to clean up temporary files
clean_up() {
    log_info "Cleaning up temporary files..."
    
    # Clean cache
    if [[ -d "cache" ]]; then
        rm -rf cache/*
        log_info "Cache cleaned"
    fi
    
    # Clean temporary files
    if [[ -d "temp" ]]; then
        rm -rf temp/*
        log_info "Temporary files cleaned"
    fi
    
    # Clean old logs (keep last 10)
    if [[ -d "logs" ]]; then
        find logs -name "*.log.*" -type f | sort -r | tail -n +11 | xargs rm -f
        log_info "Old logs cleaned"
    fi
    
    # Clean old exports (older than 30 days)
    if [[ -d "exports" ]]; then
        find exports -name "*" -type f -mtime +30 -delete
        log_info "Old exports cleaned"
    fi
    
    log_success "Cleanup completed"
}

# Main execution
case $COMMAND in
    install)
        log_info "Installing $APP_NAME for $ENVIRONMENT environment..."
        check_python
        check_pip
        setup_venv
        install_dependencies
        create_directories
        init_database
        log_success "Installation completed"
        ;;
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        sleep 2
        start_app
        ;;
    status)
        check_status
        ;;
    backup)
        create_backup
        ;;
    restore)
        restore_backup
        ;;
    update)
        update_app
        ;;
    logs)
        show_logs
        ;;
    clean)
        clean_up
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
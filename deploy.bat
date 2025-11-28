@echo off
REM Deployment Script for Lottery Prediction System (Windows)
REM This script handles deployment across different environments on Windows

setlocal enabledelayedexpansion

REM Configuration
set "APP_NAME=lottery-prediction-system"
set "DEFAULT_ENV=production"
set "DEFAULT_PORT=8501"

REM Get version from constants file
for /f "tokens=2 delims='" %%a in ('findstr "APP_VERSION" streamlit_app\configs\constants.py') do set "VERSION=%%a"

REM Initialize variables
set "ENVIRONMENT=%DEFAULT_ENV%"
set "PORT=%DEFAULT_PORT%"
set "VERBOSE=false"
set "NO_BACKUP=false"
set "FORCE=false"
set "COMMAND="

REM Color codes for Windows (limited support)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :validate_args
if /i "%~1"=="-e" (
    set "ENVIRONMENT=%~2"
    shift /1
    shift /1
    goto :parse_args
)
if /i "%~1"=="--env" (
    set "ENVIRONMENT=%~2"
    shift /1
    shift /1
    goto :parse_args
)
if /i "%~1"=="-p" (
    set "PORT=%~2"
    shift /1
    shift /1
    goto :parse_args
)
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift /1
    shift /1
    goto :parse_args
)
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-v" (
    set "VERBOSE=true"
    shift /1
    goto :parse_args
)
if /i "%~1"=="--verbose" (
    set "VERBOSE=true"
    shift /1
    goto :parse_args
)
if /i "%~1"=="--no-backup" (
    set "NO_BACKUP=true"
    shift /1
    goto :parse_args
)
if /i "%~1"=="--force" (
    set "FORCE=true"
    shift /1
    goto :parse_args
)
if /i "%~1"=="install" (
    set "COMMAND=install"
    shift /1
    goto :parse_args
)
if /i "%~1"=="start" (
    set "COMMAND=start"
    shift /1
    goto :parse_args
)
if /i "%~1"=="stop" (
    set "COMMAND=stop"
    shift /1
    goto :parse_args
)
if /i "%~1"=="restart" (
    set "COMMAND=restart"
    shift /1
    goto :parse_args
)
if /i "%~1"=="status" (
    set "COMMAND=status"
    shift /1
    goto :parse_args
)
if /i "%~1"=="backup" (
    set "COMMAND=backup"
    shift /1
    goto :parse_args
)
if /i "%~1"=="restore" (
    set "COMMAND=restore"
    shift /1
    goto :parse_args
)
if /i "%~1"=="update" (
    set "COMMAND=update"
    shift /1
    goto :parse_args
)
if /i "%~1"=="logs" (
    set "COMMAND=logs"
    shift /1
    goto :parse_args
)
if /i "%~1"=="clean" (
    set "COMMAND=clean"
    shift /1
    goto :parse_args
)
echo %RED%[ERROR]%NC% Unknown option: %~1
goto :show_help

:validate_args
if "%COMMAND%"=="" (
    echo %RED%[ERROR]%NC% No command specified
    goto :show_help
)

REM Validate environment
if not "%ENVIRONMENT%"=="development" if not "%ENVIRONMENT%"=="staging" if not "%ENVIRONMENT%"=="production" if not "%ENVIRONMENT%"=="testing" (
    echo %RED%[ERROR]%NC% Invalid environment: %ENVIRONMENT%
    exit /b 1
)

REM Set environment variables
set "LOTTERY_ENV=%ENVIRONMENT%"
set "LOTTERY_PORT=%PORT%"

goto :main

:show_help
echo Lottery Prediction System Deployment Script (Windows)
echo.
echo Usage: %0 [OPTIONS] [COMMAND]
echo.
echo Commands:
echo   install     Install dependencies and setup environment
echo   start       Start the application
echo   stop        Stop the application
echo   restart     Restart the application
echo   status      Check application status
echo   backup      Create backup of data
echo   restore     Restore from backup
echo   update      Update application
echo   logs        Show application logs
echo   clean       Clean up temporary files
echo.
echo Options:
echo   -e, --env ENV       Environment (development^|staging^|production^) [default: production]
echo   -p, --port PORT     Port number [default: 8501]
echo   -h, --help          Show this help message
echo   -v, --verbose       Verbose output
echo   --no-backup         Skip backup during update
echo   --force             Force operation without confirmation
echo.
echo Examples:
echo   %0 install                          # Install for production
echo   %0 -e development start             # Start in development mode
echo   %0 -p 8502 start                   # Start on custom port
echo   %0 backup                          # Create backup
echo   %0 update --no-backup              # Update without backup
echo.
exit /b 0

:log_info
echo %BLUE%[INFO]%NC% %~1
exit /b 0

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
exit /b 0

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
exit /b 0

:log_error
echo %RED%[ERROR]%NC% %~1
exit /b 0

:check_python
python --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python is not installed or not in PATH"
    exit /b 1
)
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%a"
call :log_info "Using Python %PYTHON_VERSION%"
exit /b 0

:check_pip
pip --version >nul 2>&1
if errorlevel 1 (
    call :log_error "pip is not installed or not in PATH"
    exit /b 1
)
exit /b 0

:setup_venv
call :log_info "Setting up virtual environment..."
if not exist "venv" (
    python -m venv venv
    call :log_success "Virtual environment created"
) else (
    call :log_info "Virtual environment already exists"
)

REM Activate virtual environment
call venv\Scripts\activate.bat
call :log_success "Virtual environment activated"
exit /b 0

:install_dependencies
call :log_info "Installing dependencies..."
if "%VERBOSE%"=="true" (
    pip install -r requirements.txt
) else (
    pip install -r requirements.txt --quiet
)
if errorlevel 1 (
    call :log_error "Failed to install dependencies"
    exit /b 1
)
call :log_success "Dependencies installed"
exit /b 0

:create_directories
call :log_info "Creating necessary directories..."
for %%d in (data logs cache exports models temp backups) do (
    if not exist "%%d" mkdir "%%d"
    if "%VERBOSE%"=="true" call :log_info "Created directory: %%d"
)
call :log_success "Directories created"
exit /b 0

:init_database
call :log_info "Initializing database..."
if exist "scripts\init_db.py" (
    python scripts\init_db.py
    call :log_success "Database initialized"
) else (
    call :log_warning "Database initialization script not found"
)
exit /b 0

:start_app
call :log_info "Starting %APP_NAME% on port %PORT% in %ENVIRONMENT% mode..."

REM Check if already running
tasklist /fi "imagename eq python.exe" /fo csv | findstr /i "streamlit" >nul
if not errorlevel 1 (
    call :log_warning "Application may already be running"
    if not "%FORCE%"=="true" (
        call :log_error "Use --force to override or stop the application first"
        exit /b 1
    )
)

REM Set additional environment variables
set "STREAMLIT_SERVER_PORT=%PORT%"
set "STREAMLIT_SERVER_ADDRESS=0.0.0.0"

if "%ENVIRONMENT%"=="production" (
    set "STREAMLIT_SERVER_HEADLESS=true"
    set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
)

REM Start the application
if "%ENVIRONMENT%"=="development" (
    streamlit run streamlit_app\app.py --server.port=%PORT%
) else (
    start /b streamlit run streamlit_app\app.py --server.port=%PORT% > logs\app.log 2>&1
    call :log_success "Application started in background"
)
exit /b 0

:stop_app
call :log_info "Stopping %APP_NAME%..."

REM Kill streamlit processes
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo csv ^| findstr /i "streamlit"') do (
    taskkill /pid %%a /f >nul 2>&1
)
call :log_success "Application stopped"
exit /b 0

:check_status
call :log_info "Checking application status..."

tasklist /fi "imagename eq python.exe" /fo csv | findstr /i "streamlit" >nul
if not errorlevel 1 (
    call :log_success "Application appears to be running"
    
    REM Check if port is responding using PowerShell
    powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:%PORT%' -UseBasicParsing -TimeoutSec 5 | Out-Null; Write-Host 'Port responding' } catch { Write-Host 'Port not responding' }" >nul 2>&1
    if not errorlevel 1 (
        call :log_success "Application is responding on port %PORT%"
    ) else (
        call :log_warning "Application may be running but not responding on port %PORT%"
    )
) else (
    call :log_info "Application is not running"
)
exit /b 0

:create_backup
call :log_info "Creating backup..."

REM Generate timestamp
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set "DATE=%%a%%b%%c"
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set "TIME=%%a%%b"
set "TIMESTAMP=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%"
set "BACKUP_DIR=backups\backup_%TIMESTAMP%"

mkdir "%BACKUP_DIR%"

REM Backup data directory
if exist "data" (
    xcopy /e /i "data" "%BACKUP_DIR%\data" >nul
    call :log_info "Data backed up"
)

REM Backup configuration
if exist "streamlit_app\configs" (
    xcopy /e /i "streamlit_app\configs" "%BACKUP_DIR%\configs" >nul
    call :log_info "Configuration backed up"
)

REM Backup models
if exist "models" (
    xcopy /e /i "models" "%BACKUP_DIR%\models" >nul
    call :log_info "Models backed up"
)

REM Create backup manifest
echo Backup created: %date% %time% > "%BACKUP_DIR%\manifest.txt"
echo Environment: %ENVIRONMENT% >> "%BACKUP_DIR%\manifest.txt"
echo Version: %VERSION% >> "%BACKUP_DIR%\manifest.txt"

call :log_success "Backup created: %BACKUP_DIR%"
exit /b 0

:restore_backup
call :log_info "Available backups:"
dir backups\backup_* /b
echo.
set /p "BACKUP_NAME=Enter backup directory name (or 'latest' for most recent): "

if /i "%BACKUP_NAME%"=="latest" (
    for /f %%i in ('dir backups\backup_* /b /o-d') do (
        set "BACKUP_NAME=%%i"
        goto :restore_continue
    )
)

:restore_continue
set "BACKUP_PATH=backups\%BACKUP_NAME%"

if not exist "%BACKUP_PATH%" (
    call :log_error "Backup not found: %BACKUP_PATH%"
    exit /b 1
)

if not "%FORCE%"=="true" (
    set /p "CONFIRM=This will overwrite current data. Continue? (y/N): "
    if /i not "!CONFIRM!"=="y" (
        call :log_info "Restore cancelled"
        exit /b 0
    )
)

call :log_info "Restoring from %BACKUP_PATH%..."

REM Stop application first
call :stop_app

REM Restore data
if exist "%BACKUP_PATH%\data" (
    if exist "data" rmdir /s /q "data"
    xcopy /e /i "%BACKUP_PATH%\data" "data" >nul
    call :log_info "Data restored"
)

REM Restore configuration
if exist "%BACKUP_PATH%\configs" (
    if exist "streamlit_app\configs" rmdir /s /q "streamlit_app\configs"
    xcopy /e /i "%BACKUP_PATH%\configs" "streamlit_app\configs" >nul
    call :log_info "Configuration restored"
)

REM Restore models
if exist "%BACKUP_PATH%\models" (
    if exist "models" rmdir /s /q "models"
    xcopy /e /i "%BACKUP_PATH%\models" "models" >nul
    call :log_info "Models restored"
)

call :log_success "Restore completed"
exit /b 0

:update_app
call :log_info "Updating %APP_NAME%..."

REM Create backup unless disabled
if not "%NO_BACKUP%"=="true" call :create_backup

REM Update dependencies
call :install_dependencies

REM Restart application
call :stop_app
timeout /t 2 /nobreak >nul
call :start_app

call :log_success "Update completed"
exit /b 0

:show_logs
call :log_info "Application logs:"
if exist "logs\app.log" (
    type "logs\app.log"
) else (
    call :log_warning "No log file found"
)
exit /b 0

:clean_up
call :log_info "Cleaning up temporary files..."

REM Clean cache
if exist "cache" (
    del /q "cache\*" >nul 2>&1
    call :log_info "Cache cleaned"
)

REM Clean temporary files
if exist "temp" (
    del /q "temp\*" >nul 2>&1
    call :log_info "Temporary files cleaned"
)

REM Clean old logs (keep last 10)
if exist "logs" (
    for /f "skip=10 delims=" %%f in ('dir /b /o-d logs\*.log.*') do del "logs\%%f" >nul 2>&1
    call :log_info "Old logs cleaned"
)

REM Clean old exports (older than 30 days)
if exist "exports" (
    forfiles /p exports /s /c "cmd /c if @isdir==FALSE del @path" /d -30 >nul 2>&1
    call :log_info "Old exports cleaned"
)

call :log_success "Cleanup completed"
exit /b 0

:main
REM Main execution
if /i "%COMMAND%"=="install" (
    call :log_info "Installing %APP_NAME% for %ENVIRONMENT% environment..."
    call :check_python
    if errorlevel 1 exit /b 1
    call :check_pip
    if errorlevel 1 exit /b 1
    call :setup_venv
    if errorlevel 1 exit /b 1
    call :install_dependencies
    if errorlevel 1 exit /b 1
    call :create_directories
    if errorlevel 1 exit /b 1
    call :init_database
    call :log_success "Installation completed"
) else if /i "%COMMAND%"=="start" (
    call :start_app
) else if /i "%COMMAND%"=="stop" (
    call :stop_app
) else if /i "%COMMAND%"=="restart" (
    call :stop_app
    timeout /t 2 /nobreak >nul
    call :start_app
) else if /i "%COMMAND%"=="status" (
    call :check_status
) else if /i "%COMMAND%"=="backup" (
    call :create_backup
) else if /i "%COMMAND%"=="restore" (
    call :restore_backup
) else if /i "%COMMAND%"=="update" (
    call :update_app
) else if /i "%COMMAND%"=="logs" (
    call :show_logs
) else if /i "%COMMAND%"=="clean" (
    call :clean_up
) else (
    call :log_error "Unknown command: %COMMAND%"
    goto :show_help
)

endlocal
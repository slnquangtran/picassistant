# Startup script for CapCut Vision

Write-Host "🚀 Starting CapCut Vision AI Ecosystem..." -ForegroundColor Cyan

# Start Backend (FastAPI)
Write-Host "--- [1/2] Launching Backend Server (FastAPI) ---" -ForegroundColor Yellow
$BackendJob = Start-Job -ScriptBlock {
    python server.py
}

# Start Frontend (Vite)
Write-Host "--- [2/2] Launching Frontend Dev Server (Vite) ---" -ForegroundColor Yellow
$FrontendJob = Start-Job -ScriptBlock {
    npm run dev
}

Write-Host "✅ Both servers are starting in the background." -ForegroundColor Green
Write-Host "Backend: http://localhost:8000"
Write-Host "Frontend: http://localhost:5173 (check terminal for exact URL)"
Write-Host ""
Write-Host "Press Ctrl+C to stop this script (and the background jobs)." -ForegroundColor Red

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "`nStopping background jobs..." -ForegroundColor Gray
    Stop-Job $BackendJob
    Stop-Job $FrontendJob
    Remove-Job $BackendJob
    Remove-Job $FrontendJob
}
